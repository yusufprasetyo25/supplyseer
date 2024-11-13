"""
hardware/sensor_reader.py
------------------------
Handles raw data reading from velostat sensor array.
"""

import numpy as np
from typing import Optional, List, Dict
import logging
from dataclasses import dataclass
import time
from queue import Queue
import threading

@dataclass
class SensorConfig:
    rows: int
    cols: int
    sampling_rate: int
    resolution: int
    reference_voltage: float
    buffer_size: int = 1000
    read_timeout: float = 1.0

@dataclass
class SensorReading:
    """Container for sensor readings"""
    data: np.ndarray
    timestamp: float
    metadata: Dict = None

class SensorReader:
    """Manages continuous sensor data reading"""
    
    def __init__(self, arduino_handler: 'ArduinoHandler', config: SensorConfig):
        self.arduino = arduino_handler
        self.config = config
        self.buffer = Queue(maxsize=config.buffer_size)
        self.is_reading = False
        self.logger = logging.getLogger(__name__)
        self._read_thread = None
        self.last_reading_time = 0
        
    def start_reading(self):
        """Start continuous reading in background thread"""
        if not self.is_reading:
            self.is_reading = True
            self._read_thread = threading.Thread(
                target=self._reading_loop,
                daemon=True
            )
            self._read_thread.start()
            self.logger.info("Started sensor reading")
            
    def stop_reading(self):
        """Stop the reading loop"""
        self.is_reading = False
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)
            self.logger.info("Stopped sensor reading")
            
    def read_frame(self) -> Optional[SensorReading]:
        """Read a single frame from the buffer"""
        try:
            return self.buffer.get(timeout=self.config.read_timeout)
        except Queue.Empty:
            return None
            
    def _reading_loop(self):
        """Continuous reading loop"""
        while self.is_reading:
            try:
                # Calculate time to next reading
                elapsed = time.time() - self.last_reading_time
                sleep_time = max(0, 1.0/self.config.sampling_rate - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Read data
                raw_data = self.arduino.read_raw_data()
                current_time = time.time()
                
                if raw_data is not None:
                    reading = SensorReading(
                        data=self._process_raw_data(raw_data),
                        timestamp=current_time,
                        metadata={
                            'sample_rate': self.config.sampling_rate,
                            'resolution': self.config.resolution
                        }
                    )
                    
                    if not self.buffer.full():
                        self.buffer.put(reading)
                    else:
                        self.logger.warning("Buffer full, dropping oldest reading")
                        self.buffer.get()
                        self.buffer.put(reading)
                        
                    self.last_reading_time = current_time
                    
            except Exception as e:
                self.logger.error(f"Error in reading loop: {e}")
                time.sleep(0.1)  # Prevent tight loop on error
                
    def _process_raw_data(self, raw_data: bytes) -> np.ndarray:
        """Process raw byte data into numpy array"""
        try:
            # Convert bytes to 16-bit integers
            values = np.frombuffer(raw_data, dtype=np.uint16)
            
            # Reshape to sensor dimensions
            matrix = values.reshape((self.config.rows, self.config.cols))
            
            # Convert to voltage values
            voltage_matrix = matrix * (self.config.reference_voltage / (2**self.config.resolution))
            
            return voltage_matrix
            
        except Exception as e:
            self.logger.error(f"Data processing error: {e}")
            raise
