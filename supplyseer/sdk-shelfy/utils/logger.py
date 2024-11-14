"""
utils/logger.py
--------------
Logging system with advanced features.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Dict
import json
from datetime import datetime
import queue
from threading import Lock

class LogManager:
    """Advanced logging system manager"""
    
    def __init__(self,
                 config_path: str = 'config/logging_config.yaml',
                 log_dir: str = 'logs',
                 default_level: str = 'INFO'):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.config_path = Path(config_path)
        self.default_level = getattr(logging, default_level)
        
        # Initialize components
        self.handlers = {}
        self.formatters = {}
        self.loggers = {}
        self.message_queue = queue.Queue()
        self.handler_lock = Lock()
        
        # Setup logging system
        self._load_config()
        self._setup_logging()
        
    def _load_config(self):
        """Load logging configuration"""
        try:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading logging config: {e}")
            self.config = self._default_config()
            
    def _setup_logging(self):
        """Configure logging system"""
        # Create formatters
        self._setup_formatters()
        
        # Create handlers
        self._setup_handlers()
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config['root']['level'])
        
        # Add handlers to root logger
        for handler_name in self.config['root']['handlers']:
            if handler_name in self.handlers:
                root_logger.addHandler(self.handlers[handler_name])
                
        # Setup component loggers
        self._setup_component_loggers()
        
    def _setup_formatters(self):
        """Create log formatters"""
        for name, fmt_config in self.config['formatters'].items():
            self.formatters[name] = logging.Formatter(
                fmt_config['format'],
                datefmt=fmt_config.get('datefmt', '%Y-%m-%d %H:%M:%S')
            )
            
    def _setup_handlers(self):
        """Create log handlers"""
        for name, handler_config in self.config['handlers'].items():
            if handler_config['class'] == 'logging.StreamHandler':
                handler = logging.StreamHandler(sys.stdout)
            elif handler_config['class'] == 'logging.handlers.RotatingFileHandler':
                handler = RotatingFileHandler(
                    self.log_dir / handler_config['filename'],
                    maxBytes=handler_config['maxBytes'],
                    backupCount=handler_config['backupCount'],
                    encoding=handler_config['encoding']
                )
            elif handler_config['class'] == 'logging.handlers.TimedRotatingFileHandler':
                handler = TimedRotatingFileHandler(
                    self.log_dir / handler_config['filename'],
                    when=handler_config.get('when', 'midnight'),
                    interval=handler_config.get('interval', 1),
                    backupCount=handler_config.get('backupCount', 7),
                    encoding=handler_config.get('encoding', 'utf8')
                )
                
            # Set formatter
            formatter_name = handler_config.get('formatter', 'standard')
            if formatter_name in self.formatters:
                handler.setFormatter(self.formatters[formatter_name])
                
            # Set level
            handler.setLevel(getattr(logging, handler_config['level']))
            
            self.handlers[name] = handler
            
    def _setup_component_loggers(self):
        """Setup loggers for different components"""
        for name, logger_config in self.config['loggers'].items():
            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, logger_config['level']))
            logger.propagate = logger_config.get('propagate', True)
            
            # Add handlers
            for handler_name in logger_config['handlers']:
                if handler_name in self.handlers:
                    logger.addHandler(self.handlers[handler_name])
                    
            self.loggers[name] = logger
            
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger for specific component"""
        return logging.getLogger(name)
        
    def log_message(self, level: str, message: str, **kwargs):
        """Log message with additional context"""
        logger = self.get_logger(kwargs.get('component', 'root'))
        log_level = getattr(logging, level.upper())
        
        # Add timestamp and context
        context = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            **kwargs
        }
        
        # Format message with context
        formatted_message = f"{message} | {json.dumps(context)}"
        
        # Log message
        logger.log(log_level, formatted_message)
        
        # Store in queue for analysis
        self.message_queue.put((context, formatted_message))
        
    def analyze_logs(self, hours: int = 24) -> Dict:
        """Analyze recent logs"""
        analysis = {
            'error_count': 0,
            'warning_count': 0,
            'component_stats': {},
            'error_summary': []
        }
        
        # Analyze queued messages
        while not self.message_queue.empty():
            context, _ = self.message_queue.get()
            level = context['level'].upper()
            component = context.get('component', 'root')
            
            # Update statistics
            if level == 'ERROR':
                analysis['error_count'] += 1
                analysis['error_summary'].append(context)
            elif level == 'WARNING':
                analysis['warning_count'] += 1
                
            # Update component stats
            if component not in analysis['component_stats']:
                analysis['component_stats'][component] = {'message_count': 0}
            analysis['component_stats'][component]['message_count'] += 1
            
        return analysis