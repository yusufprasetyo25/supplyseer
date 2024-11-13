"""
hardware/arduino_handler.py
--------------------------
Handles communication with Arduino for velostat sensor data acquisition.
"""

import serial
import numpy as np
from typing import Optional, Tuple
import logging
from dataclasses import dataclass
import time

@dataclass
class ArduinoConfig:
    port: str
    baudrate: int
    timeout: float
    read_size: int = 225  # 15x15 matrix = 225 values

class ArduinoHandler:
    """Manages serial communication with Arduino"""
    
    def __init__(self, config: ArduinoConfig):
        self.config = config
        self.serial = None
        self.logger = logging.getLogger(__name__)
        self.is_connected = False
        
    def connect(self) -> bool:
        """Establish connection with Arduino"""
        try:
            self.serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout
            )
            # Wait for Arduino reset
            time.sleep(2)
            # Clear any startup messages
            self.serial.reset_input_buffer()
            
            self.is_connected = True
            self.logger.info(f"Connected to Arduino on {self.config.port}")
            return True
            
        except serial.SerialException as e:
            self.logger.error(f"Failed to connect: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Close Arduino connection"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.is_connected = False
            self.logger.info("Disconnected from Arduino")
            
    def read_raw_data(self) -> Optional[bytes]:
        """Read raw data from Arduino"""
        if not self.is_connected or not self.serial.is_open:
            raise ConnectionError("Arduino not connected")
            
        try:
            # Wait for start marker
            while True:
                if self.serial.read() == b'S':
                    break
                    
            # Read the full frame
            raw_data = self.serial.read(self.config.read_size * 2)  # 2 bytes per value
            
            if len(raw_data) == self.config.read_size * 2:
                return raw_data
            else:
                self.logger.warning("Incomplete data frame received")
                return None
                
        except serial.SerialException as e:
            self.logger.error(f"Error reading data: {e}")
            self.is_connected = False
            return None
            
    def flush_buffers(self):
        """Flush serial buffers"""
        if self.serial and self.serial.is_open:
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()