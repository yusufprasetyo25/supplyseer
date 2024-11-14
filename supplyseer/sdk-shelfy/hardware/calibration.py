"""
hardware/calibration.py
----------------------
Handles sensor calibration and normalization.
"""

import numpy as np
import time
from typing import Dict, List, Optional
import logging
import json
from pathlib import Path

@dataclass
class CalibrationConfig:
    num_samples: int
    settling_time: float
    calibration_weights: List[float]
    reference_temp: float
    temp_coefficient: float
    max_age: int
    uniformity_threshold: float
    drift_threshold: float

class Calibration:
    """Manages sensor calibration procedures"""
    
    def __init__(self, config: CalibrationConfig, sensor_config: SensorConfig):
        self.config = config
        self.sensor_config = sensor_config
        self.calibration_data = None
        self.logger = logging.getLogger(__name__)
        self.calibration_path = Path('config/calibration_data.json')
        self._load_calibration()
        
    def perform_calibration(self, sensor_reader: SensorReader) -> bool:
        """Perform full calibration procedure"""
        try:
            self.logger.info("Starting calibration procedure...")
            
            # Collect baseline readings
            baseline = self._collect_baseline(sensor_reader)
            
            # Calculate calibration factors
            factors = self._calculate_calibration_factors(baseline)
            
            # Validate calibration
            if self._validate_calibration(factors):
                self.calibration_data = {
                    'baseline': baseline.tolist(),
                    'factors': factors.tolist(),
                    'timestamp': time.time(),
                    'temperature': self.config.reference_temp
                }
                
                # Save calibration data
                self._save_calibration()
                self.logger.info("Calibration completed successfully")
                return True
            else:
                self.logger.error("Calibration validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            return False
            
    def apply_calibration(self, data: np.ndarray, temperature: float = None) -> np.ndarray:
        """Apply calibration to sensor data"""
        if self.calibration_data is None:
            self.logger.warning("No calibration data available")
            return data
            
        try:
            # Apply calibration factors
            calibrated = data * np.array(self.calibration_data['factors'])
            
            # Apply temperature compensation if available
            if temperature is not None:
                temp_diff = temperature - self.config.reference_temp
                temp_factor = 1 + (temp_diff * self.config.temp_coefficient)
                calibrated = calibrated * temp_factor
                
            return calibrated
            
        except Exception as e:
            self.logger.error(f"Error applying calibration: {e}")
            return data
            
    def _collect_baseline(self, sensor_reader: SensorReader) -> np.ndarray:
        """Collect baseline readings for calibration"""
        readings = []
        self.logger.info(f"Collecting {self.config.num_samples} baseline samples...")
        
        # Clear any existing data
        sensor_reader.buffer.queue.clear()
        
        # Wait for settling time
        time.sleep(self.config.settling_time)
        
        # Collect samples
        for _ in range(self.config.num_samples):
            reading = sensor_reader.read_frame()
            if reading is not None:
                readings.append(reading.data)
            time.sleep(1.0 / sensor_reader.config.sampling_rate)
            
        return np.mean(readings, axis=0)
        
    def _calculate_calibration_factors(self, baseline: np.ndarray) -> np.ndarray:
        """Calculate calibration factors"""
        # Prevent division by zero
        safe_baseline = np.where(baseline > 0, baseline, 1e-6)
        
        # Calculate factors
        max_value = 2 ** self.sensor_config.resolution - 1
        factors = max_value / safe_baseline
        
        return factors
        
    def _validate_calibration(self, factors: np.ndarray) -> bool:
        """Validate calibration factors"""
        # Check uniformity
        uniformity = np.std(factors) / np.mean(factors)
        if uniformity > self.config.uniformity_threshold:
            self.logger.warning(f"Calibration uniformity ({uniformity}) exceeds threshold")
            return False
            
        # Check for extreme values
        if np.any(factors > 10) or np.any(factors < 0.1):
            self.logger.warning("Calibration factors outside reasonable range")
            return False
            
        return True
        
    def _save_calibration(self):
        """Save calibration data to file"""
        try:
            with open(self.calibration_path, 'w') as f:
                json.dump(self.calibration_data, f)
            self.logger.info(f"Calibration data saved to {self.calibration_path}")
        except Exception as e:
            self.logger.error(f"Error saving calibration data: {e}")
            
    def _load_calibration(self):
        """Load calibration data from file"""
        try:
            if self.calibration_path.exists():
                with open(self.calibration_path) as f:
                    self.calibration_data = json.load(f)
                self.logger.info("Loaded existing calibration data")
        except Exception as e:
            self.logger.error(f"Error loading calibration data: {e}")