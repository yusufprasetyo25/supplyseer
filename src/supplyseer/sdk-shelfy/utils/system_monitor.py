"""
utils/system_monitor.py
----------------------
System monitoring and diagnostics.
"""

import psutil
import time
from typing import Dict, List, Optional, Callable
import logging
from dataclasses import dataclass
import threading
import numpy as np
from datetime import datetime, timedelta
from collections import deque

@dataclass
class SystemStats:
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    processing_time: float
    buffer_usage: float
    temperature: Optional[float] = None
    frame_rate: Optional[float] = None
    network_usage: Optional[Dict] = None

@dataclass
class MonitoringConfig:
    enabled: bool
    interval: int
    alerts: Dict[str, float]
    history_length: int = 3600  # 1 hour at 1-second intervals
    alert_cooldown: int = 300   # 5 minutes between repeated alerts
    log_to_file: bool = True

class SystemMonitor:
    """Monitors system performance and resources"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.stats_history = deque(maxlen=self.config.history_length)
        self.start_time = time.time()
        self.is_monitoring = False
        self._monitor_thread = None
        self.alert_callbacks: List[Callable] = []
        self.last_alert_times: Dict[str, float] = {}
        self.network_stats_prev = psutil.net_io_counters()
        self.last_stats_time = time.time()
        self.frame_count = 0
        self.frame_times = deque(maxlen=100)
        
    def start_monitoring(self):
        """Start system monitoring"""
        if not self.config.enabled:
            return
            
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("System monitoring started")
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        self.logger.info("System monitoring stopped")
        
        if self.config.log_to_file:
            self._save_monitoring_data()
        
    def add_alert_callback(self, callback: Callable):
        """Add callback for system alerts"""
        self.alert_callbacks.append(callback)
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect stats
                stats = self.collect_stats()
                self._update_history(stats)
                
                # Check thresholds and trigger alerts if needed
                self._check_thresholds(stats)
                
                # Wait for next interval
                time.sleep(self.config.interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(1.0)
                
    def collect_stats(self) -> SystemStats:
        """Collect current system statistics"""
        try:
            current_time = time.time()
            
            # Calculate frame rate
            if self.frame_times:
                frame_rate = len(self.frame_times) / (current_time - self.frame_times[0])
            else:
                frame_rate = 0
                
            # Get network usage
            network_stats = psutil.net_io_counters()
            time_diff = current_time - self.last_stats_time
            
            network_usage = {
                'bytes_sent': (network_stats.bytes_sent - self.network_stats_prev.bytes_sent) / time_diff,
                'bytes_recv': (network_stats.bytes_recv - self.network_stats_prev.bytes_recv) / time_diff
            }
            
            self.network_stats_prev = network_stats
            self.last_stats_time = current_time
            
            stats = SystemStats(
                timestamp=current_time,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                disk_usage=psutil.disk_usage('/').percent,
                processing_time=current_time - self.start_time,
                buffer_usage=self._get_buffer_usage(),
                frame_rate=frame_rate,
                network_usage=network_usage,
                temperature=self._get_cpu_temperature()
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error collecting system stats: {e}")
            raise
            
    def _get_buffer_usage(self) -> float:
        """Get buffer usage percentage"""
        try:
            virtual_memory = psutil.virtual_memory()
            return (virtual_memory.total - virtual_memory.available) / virtual_memory.total * 100
        except:
            return 0.0
            
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature if available"""
        try:
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Try different temperature sources based on platform
                    for source in ['coretemp', 'k10temp', 'cpu_thermal']:
                        if source in temps:
                            return temps[source][0].current
            return None
        except:
            return None
            
    def _update_history(self, stats: SystemStats):
        """Update monitoring history"""
        self.stats_history.append(stats)
        
    def _check_thresholds(self, stats: SystemStats):
        """Check if any monitoring thresholds are exceeded"""
        current_time = time.time()
        
        for metric, threshold in self.config.alerts.items():
            value = getattr(stats, metric, None)
            if value is not None and value > threshold:
                # Check alert cooldown
                last_alert = self.last_alert_times.get(metric, 0)
                if current_time - last_alert >= self.config.alert_cooldown:
                    self._trigger_alert(metric, value, threshold)
                    self.last_alert_times[metric] = current_time
                    
    def _trigger_alert(self, metric: str, value: float, threshold: float):
        """Trigger alert callbacks"""
        alert_message = f"System alert: {metric} at {value:.1f}% (threshold: {threshold}%)"
        self.logger.warning(alert_message)
        
        for callback in self.alert_callbacks:
            try:
                callback(metric, value, threshold)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
                
    def register_frame(self):
        """Register a frame for FPS calculation"""
        self.frame_times.append(time.time())
        self.frame_count += 1
        
    def get_performance_report(self) -> Dict:
        """Generate performance report from collected statistics"""
        if not self.stats_history:
            return {}
            
        stats_array = np.array([
            (s.cpu_usage, s.memory_usage, s.disk_usage)
            for s in self.stats_history
        ])
        
        return {
            'avg_cpu_usage': np.mean(stats_array[:, 0]),
            'max_cpu_usage': np.max(stats_array[:, 0]),
            'avg_memory_usage': np.mean(stats_array[:, 1]),
            'max_memory_usage': np.max(stats_array[:, 1]),
            'avg_disk_usage': np.mean(stats_array[:, 2]),
            'max_disk_usage': np.max(stats_array[:, 2]),
            'avg_frame_rate': np.mean([s.frame_rate for s in self.stats_history if s.frame_rate]),
            'total_frames': self.frame_count,
            'total_runtime': time.time() - self.start_time
        }
        
    def _save_monitoring_data(self):
        """Save monitoring data to file"""
        try:
            filename = f"monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w') as f:
                # Write header
                f.write("timestamp,cpu_usage,memory_usage,disk_usage,frame_rate,temperature\n")
                
                # Write data
                for stats in self.stats_history:
                    f.write(f"{stats.timestamp},{stats.cpu_usage},{stats.memory_usage},"
                           f"{stats.disk_usage},{stats.frame_rate},{stats.temperature}\n")
                           
            self.logger.info(f"Monitoring data saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving monitoring data: {e}")
            
    def check_system_health(self) -> Dict[str, bool]:
        """Check overall system health"""
        if not self.stats_history:
            return {'healthy': False, 'reason': 'No monitoring data available'}
            
        latest_stats = self.stats_history[-1]
        health_checks = {
            'cpu': latest_stats.cpu_usage < self.config.alerts['cpu_usage'],
            'memory': latest_stats.memory_usage < self.config.alerts['memory_usage'],
            'disk': latest_stats.disk_usage < self.config.alerts['disk_usage'],
            'frame_rate': latest_stats.frame_rate > 10 if latest_stats.frame_rate else False
        }
        
        is_healthy = all(health_checks.values())
        
        return {
            'healthy': is_healthy,
            'checks': health_checks,
            'timestamp': latest_stats.timestamp
        }