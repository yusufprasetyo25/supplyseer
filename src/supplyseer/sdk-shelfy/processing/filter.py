"""
processing/filter.py
-------------------
Signal filtering and noise reduction.
"""

from scipy import signal
from scipy import ndimage
import numpy as np
from typing import Optional, Dict
import logging
from dataclasses import dataclass

@dataclass
class FilterConfig:
    # Median filter
    median_kernel_size: int
    
    # Gaussian filter
    gaussian_sigma: float
    
    # Butterworth filter
    butterworth_cutoff: float
    butterworth_order: int
    
    # General settings
    enable_temporal: bool = True
    temporal_window: int = 5

class SignalFilter:
    """Handles signal filtering and noise reduction"""
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.history = []
        self._init_butterworth()
        
    def _init_butterworth(self):
        """Initialize Butterworth filter coefficients"""
        self.b, self.a = signal.butter(
            self.config.butterworth_order,
            self.config.butterworth_cutoff,
            fs=1.0  # Normalized frequency
        )
        
    def apply_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply all filters to data"""
        try:
            filtered = data.copy()
            
            # Apply spatial filters
            filtered = self._apply_spatial_filters(filtered)
            
            # Apply temporal filtering if enabled
            if self.config.enable_temporal:
                filtered = self._apply_temporal_filters(filtered)
                
            return filtered
            
        except Exception as e:
            self.logger.error(f"Filtering error: {e}")
            raise
            
    def _apply_spatial_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply spatial domain filters"""
        # Apply median filter
        filtered = signal.medfilt2d(
            data,
            kernel_size=self.config.median_kernel_size
        )
        
        # Apply Gaussian filter
        filtered = ndimage.gaussian_filter(
            filtered,
            sigma=self.config.gaussian_sigma
        )
        
        return filtered
        
    def _apply_temporal_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply temporal domain filters"""
        # Update history
        self.history.append(data)
        if len(self.history) > self.config.temporal_window:
            self.history.pop(0)
            
        # Apply temporal filtering only if we have enough history
        if len(self.history) == self.config.temporal_window:
            # Stack history into 3D array
            temporal_data = np.stack(self.history, axis=0)
            
            # Apply Butterworth filter along temporal axis
            filtered = signal.filtfilt(
                self.b, self.a,
                temporal_data,
                axis=0
            )
            
            # Return latest filtered frame
            return filtered[-1]
            
        return data
        
    def reset(self):
        """Reset filter state"""
        self.history.clear()
        self._init_butterworth()