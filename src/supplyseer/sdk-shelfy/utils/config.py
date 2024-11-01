"""
utils/config.py
--------------
Configuration management system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
import json
import os
from copy import deepcopy

@dataclass
class ArduinoConfig:
    port: str
    baudrate: int
    timeout: float

@dataclass
class SensorConfig:
    rows: int
    cols: int
    sampling_rate: int
    resolution: int
    reference_voltage: float

@dataclass
class ProcessingConfig:
    median_kernel_size: int
    gaussian_sigma: float
    butterworth_cutoff: float
    butterworth_order: int
    thresholds: Dict[str, float]
    matrix: Dict[str, bool]

@dataclass
class VisualizationConfig:
    update_interval: int
    window_size: tuple
    colormap: str
    plots: Dict[str, dict]

@dataclass
class ExportConfig:
    format: str
    compression: bool
    auto_export: bool
    export_interval: int
    directories: Dict[str, str]

class ConfigManager:
    """Manages system configuration and settings"""
    
    def __init__(self, config_path: str = 'config/settings.yaml'):
        self.config_path = Path(config_path)
        self.config = None
        self.logger = logging.getLogger(__name__)
        self.load_config()
        
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            self._validate_config()
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.warning("Loading default configuration")
            self.config = self._default_config()
            
    def _validate_config(self):
        """Validate configuration values"""
        required_sections = ['hardware', 'processing', 'visualization', 'export', 'system']
        
        # Check required sections
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
                
        # Validate specific values
        self._validate_hardware_config()
        self._validate_processing_config()
        self._validate_visualization_config()
        
    def _validate_hardware_config(self):
        """Validate hardware configuration"""
        hw_config = self.config['hardware']
        
        # Validate Arduino settings
        arduino = hw_config.get('arduino', {})
        if not arduino.get('port'):
            # Auto-detect port if not specified
            arduino['port'] = self._detect_arduino_port()
            
        # Validate sensor settings
        sensor = hw_config.get('sensor', {})
        if sensor.get('rows', 0) * sensor.get('cols', 0) == 0:
            raise ValueError("Invalid sensor dimensions")
            
    def _detect_arduino_port(self) -> str:
        """Auto-detect Arduino port"""
        # Implementation depends on platform
        if os.name == 'nt':  # Windows
            return 'COM3'
        return '/dev/ttyUSB0'  # Linux default
        
    def get_arduino_config(self) -> ArduinoConfig:
        """Get Arduino-specific configuration"""
        arduino_config = self.config['hardware']['arduino']
        return ArduinoConfig(**arduino_config)
        
    def get_sensor_config(self) -> SensorConfig:
        """Get sensor-specific configuration"""
        sensor_config = self.config['hardware']['sensor']
        return SensorConfig(**sensor_config)
        
    def get_processing_config(self) -> ProcessingConfig:
        """Get processing-specific configuration"""
        proc_config = self.config['processing']
        return ProcessingConfig(
            median_kernel_size=proc_config['filter']['median_kernel_size'],
            gaussian_sigma=proc_config['filter']['gaussian_sigma'],
            butterworth_cutoff=proc_config['filter']['butterworth']['cutoff'],
            butterworth_order=proc_config['filter']['butterworth']['order'],
            thresholds=proc_config['thresholds'],
            matrix=proc_config['matrix']
        )
        
    def get_visualization_config(self) -> VisualizationConfig:
        """Get visualization-specific configuration"""
        viz_config = self.config['visualization']
        return VisualizationConfig(**viz_config['display'])
        
    def get_export_config(self) -> ExportConfig:
        """Get export-specific configuration"""
        export_config = self.config['export']
        return ExportConfig(**export_config)
        
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
            
    def update_config(self, updates: Dict):
        """Update configuration with new values"""
        try:
            # Create a backup
            backup = deepcopy(self.config)
            
            # Apply updates
            self._recursive_update(self.config, updates)
            
            # Validate new configuration
            self._validate_config()
            
            # Save updated configuration
            self.save_config()
            
        except Exception as e:
            self.logger.error(f"Configuration update failed: {e}")
            self.config = backup
            raise
            
    def _recursive_update(self, d: Dict, u: Dict):
        """Recursively update dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                self._recursive_update(d[k], v)
            else:
                d[k] = v