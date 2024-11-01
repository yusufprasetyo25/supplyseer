"""
system_init.py
-------------
Main system initialization and integration
"""

from pathlib import Path
import logging
from typing import Dict, Optional

from utils.config import ConfigManager
from utils.logger import LogManager
from hardware.arduino_handler import ArduinoHandler
from hardware.sensor_reader import SensorReader
from hardware.calibration import Calibration
from processing.data_processor import DataProcessor
from processing.matrix_handler import MatrixHandler
from processing.filter import SignalFilter
from visualization.pressure_map import PressureVisualizer
from visualization.real_time_plot import RealTimePlotter
from visualization.data_export import DataExporter
from utils.system_monitor import SystemMonitor

class VelostatSystem:
    """Main system class that initializes all components with proper configuration"""
    
    def __init__(self, settings_path: str = 'config/settings.yaml',
                 calibration_path: str = 'config/calibration.yaml',
                 logging_path: str = 'config/logging_config.yaml'):
        # Initialize configuration managers
        self.config_manager = ConfigManager(settings_path)
        self.calibration_manager = ConfigManager(calibration_path)
        
        # Initialize logging first
        self.log_manager = LogManager(logging_path)
        self.logger = self.log_manager.get_logger(__name__)
        
        # Initialize system components
        self.init_components()
        
    def init_components(self):
        """Initialize all system components with their respective configurations"""
        try:
            # Get all required configurations
            arduino_config = self.config_manager.get_arduino_config()
            sensor_config = self.config_manager.get_sensor_config()
            processing_config = self.config_manager.get_processing_config()
            visualization_config = self.config_manager.get_visualization_config()
            
            # Hardware initialization
            self.arduino = ArduinoHandler(arduino_config)
            
            self.sensor = SensorReader(
                arduino=self.arduino,
                config=sensor_config
            )
            
            self.calibration = Calibration(
                config=self.calibration_manager.get_calibration_config(),
                sensor_config=sensor_config
            )
            
            # Processing initialization
            self.processor = DataProcessor(processing_config)
            
            self.matrix_handler = MatrixHandler(
                config=processing_config.matrix_config
            )
            
            self.signal_filter = SignalFilter(
                config=processing_config.filter_config
            )
            
            # Visualization initialization
            self.visualizer = PressureVisualizer(visualization_config)
            
            self.plotter = RealTimePlotter(
                config=visualization_config.plot_config
            )
            
            # Data export initialization
            self.exporter = DataExporter(
                config=self.config_manager.get_export_config()
            )
            
            # System monitoring initialization
            self.monitor = SystemMonitor(
                config=self.config_manager.get_monitoring_config()
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise

    def start(self):
        """Start the system with all components"""
        try:
            # Connect to hardware
            if not self.arduino.connect():
                raise ConnectionError("Failed to connect to Arduino")
            
            # Perform initial calibration if needed
            if self.calibration_manager.config['calibration']['perform_initial_calibration']:
                self.calibration.perform_calibration(self.sensor)
            
            # Start data acquisition
            self.sensor.start_reading()
            
            # Start visualization if enabled
            if self.config_manager.config['visualization']['display']['enabled']:
                self.visualizer.setup_plot()
                self.plotter.setup_plots()
                self.visualizer.start_animation(self._data_generator())
            
            # Start system monitoring
            if self.config_manager.config['system']['monitoring']['enabled']:
                self.monitor.start_monitoring()
            
            self.logger.info("System started successfully")
            
        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            raise
            
    def _data_generator(self):
        """Generate processed data for visualization"""
        while True:
            try:
                # Get raw data
                raw_data = self.sensor.read_frame()
                if raw_data is None:
                    continue
                
                # Process data pipeline
                filtered_data = self.signal_filter.apply_filters(raw_data)
                processed_data = self.processor.process_frame(filtered_data)
                matrix_data = self.matrix_handler.process_matrix(processed_data)
                
                # Handle data export if enabled
                if self.config_manager.config['export']['auto_export']:
                    self.exporter.export_frame({
                        'raw': raw_data,
                        'processed': processed_data,
                        'matrix': matrix_data
                    })
                
                # Update system monitoring
                self.monitor.update_processing_stats(processed_data)
                
                yield {
                    'pressure': processed_data['processed'],
                    'metrics': processed_data['metrics'],
                    'matrix_data': matrix_data,
                    'timestamp': raw_data.timestamp
                }
                
            except Exception as e:
                self.logger.error(f"Data processing error: {e}")
                continue

    def stop(self):
        """Safely stop all system components"""
        try:
            # Stop data acquisition
            self.sensor.stop_reading()
            
            # Stop visualization
            self.visualizer.stop_animation()
            
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            # Disconnect hardware
            self.arduino.disconnect()
            
            # Final export if needed
            if self.config_manager.config['export']['export_on_stop']:
                self.exporter.export_session(self.monitor.get_session_data())
            
            self.logger.info("System stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during system shutdown: {e}")
            raise

def main():
    """Main entry point"""
    system = None
    try:
        # Create and start system
        system = VelostatSystem()
        system.start()
        
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        if system:
            system.stop()

if __name__ == "__main__":
    main()