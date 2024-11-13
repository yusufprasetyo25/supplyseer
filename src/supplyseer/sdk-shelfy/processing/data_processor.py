"""
processing/data_processor.py
---------------------------
Handles processing of velostat sensor data.
"""

import numpy as np
from scipy import signal
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from hardware.sensor_reader import SensorReading

@dataclass
class ProcessingConfig:
    # Thresholds
    pressure_min: float
    pressure_max: float
    activity_threshold: float
    noise_threshold: float
    
    # Processing parameters
    normalize: bool = True
    subtract_baseline: bool = True
    apply_smoothing: bool = True
    
    # Feature extraction
    extract_features: bool = True
    temporal_window: int = 10  # frames

class DataProcessor:
    """Processes velostat sensor data"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.baseline = None
        self.history = []
        
    def process_frame(self, reading: SensorReading) -> Dict:
        """Process a single frame of sensor data"""
        try:
            # Initialize baseline if needed
            if self.baseline is None and self.config.subtract_baseline:
                self.baseline = reading.data
                
            # Pre-process data
            processed = self._preprocess_data(reading.data)
            
            # Extract features
            features = self._extract_features(processed) if self.config.extract_features else {}
            
            # Update history
            self._update_history(processed)
            
            return {
                'processed': processed,
                'features': features,
                'timestamp': reading.timestamp,
                'metadata': reading.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            raise
            
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess raw sensor data"""
        processed = data.copy()
        
        # Subtract baseline if enabled
        if self.config.subtract_baseline and self.baseline is not None:
            processed = processed - self.baseline
            
        # Apply thresholds
        processed = np.clip(processed, self.config.pressure_min, self.config.pressure_max)
        
        # Normalize if enabled
        if self.config.normalize:
            processed = self._normalize_data(processed)
            
        return processed
        
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize pressure values"""
        min_val = self.config.pressure_min
        max_val = self.config.pressure_max
        return (data - min_val) / (max_val - min_val + 1e-6)
        
    def _extract_features(self, data: np.ndarray) -> Dict:
        """Extract features from processed data"""
        return {
            'mean_pressure': np.mean(data),
            'max_pressure': np.max(data),
            'active_cells': np.sum(data > self.config.activity_threshold),
            'pressure_sum': np.sum(data),
            'center_of_pressure': self._calculate_cop(data),
            'activity_map': data > self.config.activity_threshold
        }
        
    def _calculate_cop(self, data: np.ndarray) -> Tuple[float, float]:
        """Calculate center of pressure"""
        total = np.sum(data)
        if total < 1e-6:
            return (0.0, 0.0)
            
        x = np.sum(data * np.arange(data.shape[1])) / total
        y = np.sum(data * np.arange(data.shape[0])[:, np.newaxis]) / total
        
        return (x, y)
        
    def _update_history(self, data: np.ndarray):
        """Update processing history"""
        self.history.append(data)
        if len(self.history) > self.config.temporal_window:
            self.history.pop(0)
