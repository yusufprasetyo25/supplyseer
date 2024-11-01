"""
processing/matrix_handler.py
---------------------------
Handles matrix operations and transformations.
"""

import numpy as np
from scipy import ndimage
from typing import Dict, Tuple, List
import logging
from dataclasses import dataclass

@dataclass
class MatrixConfig:
    interpolation: bool
    smoothing: bool
    edge_detection: bool
    region_threshold: float = 0.2
    min_region_size: int = 4
    max_regions: int = 10

class MatrixHandler:
    """Handles matrix operations for pressure data"""
    
    def __init__(self, config: MatrixConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def process_matrix(self, data: np.ndarray) -> Dict:
        """Process pressure matrix"""
        try:
            results = {}
            
            # Apply interpolation if enabled
            if self.config.interpolation:
                data = self._interpolate_matrix(data)
                results['interpolated'] = data
                
            # Calculate gradients
            gradients = self._calculate_gradients(data)
            results['gradients'] = gradients
            
            # Identify regions
            if self.config.edge_detection:
                regions = self._identify_regions(data)
                results['regions'] = regions
                
            # Extract features
            results['features'] = self._extract_features(data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Matrix processing error: {e}")
            raise
            
    def _interpolate_matrix(self, data: np.ndarray) -> np.ndarray:
        """Interpolate matrix to higher resolution"""
        return ndimage.zoom(data, 2, order=1)
        
    def _calculate_gradients(self, data: np.ndarray) -> Dict:
        """Calculate pressure gradients"""
        grad_y, grad_x = np.gradient(data)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        return {
            'magnitude': magnitude,
            'direction': direction,
            'grad_x': grad_x,
            'grad_y': grad_y
        }
        
    def _identify_regions(self, data: np.ndarray) -> Dict:
        """Identify distinct pressure regions"""
        # Threshold the data
        binary = data > self.config.region_threshold
        
        # Label connected regions
        labeled, num_features = ndimage.label(binary)
        
        # Filter small regions
        sizes = np.bincount(labeled.ravel())[1:]
        mask_sizes = sizes >= self.config.min_region_size
        valid_labels = np.arange(1, num_features + 1)[mask_sizes]
        
        # Limit number of regions
        if len(valid_labels) > self.config.max_regions:
            valid_labels = valid_labels[:self.config.max_regions]
            
        # Create filtered label matrix
        filtered_labels = np.zeros_like(labeled)
        for i, label in enumerate(valid_labels, 1):
            filtered_labels[labeled == label] = i
            
        return {
            'labels': filtered_labels,
            'num_regions': len(valid_labels),
            'region_sizes': sizes[mask_sizes],
            'region_centers': self._calculate_region_centers(filtered_labels)
        }
        
    def _calculate_region_centers(self, labeled_matrix: np.ndarray) -> List[Tuple[float, float]]:
        """Calculate center of mass for each region"""
        centers = []
        for label in range(1, np.max(labeled_matrix) + 1):
            region = labeled_matrix == label
            if np.any(region):
                center = ndimage.center_of_mass(region)
                centers.append(center)
        return centers
        
    def _extract_features(self, data: np.ndarray) -> Dict:
        """Extract matrix features"""
        return {
            'mean': np.mean(data),
            'max': np.max(data),
            'std': np.std(data),
            'total_pressure': np.sum(data),
            'active_area': np.sum(data > self.config.region_threshold) / data.size,
            'uniformity': np.std(data) / (np.mean(data) + 1e-6)
        }
