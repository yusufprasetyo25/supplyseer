"""
visualization/pressure_map.py
----------------------------
Real-time visualization of pressure data.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Callable
import logging
from dataclasses import dataclass
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors

@dataclass
class VisualizationConfig:
    update_interval: int
    colormap: str
    window_size: tuple
    interpolation: str = 'nearest'
    show_colorbar: bool = True
    show_grid: bool = True
    title: str = 'Pressure Map'
    vmin: float = 0.0
    vmax: float = 1.0

class PressureVisualizer:
    """Real-time pressure map visualization"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.fig, self.ax = plt.subplots(figsize=self.config.window_size)
        self.heatmap = None
        self.colorbar = None
        self.animation = None
        self.logger = logging.getLogger(__name__)
        self._setup_plot()
        
    def _setup_plot(self):
        """Initialize the plot with styling"""
        # Set title and labels
        self.ax.set_title(self.config.title)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        
        # Create initial heatmap
        self.heatmap = self.ax.imshow(
            np.zeros((15, 15)),
            cmap=self.config.colormap,
            interpolation=self.config.interpolation,
            vmin=self.config.vmin,
            vmax=self.config.vmax,
            aspect='equal'
        )
        
        # Add colorbar if enabled
        if self.config.show_colorbar:
            self.colorbar = plt.colorbar(self.heatmap, ax=self.ax)
            self.colorbar.set_label('Pressure')
            
        # Show grid if enabled
        if self.config.show_grid:
            self.ax.grid(True, which='minor', color='w', linestyle='-', linewidth=0.5)
            
        # Tight layout
        plt.tight_layout()
        
    def update_plot(self, frame_data: Dict):
        """Update plot with new data"""
        try:
            # Update heatmap data
            self.heatmap.set_array(frame_data['pressure'])
            
            # Update title with metadata if available
            if 'metadata' in frame_data:
                self.ax.set_title(f"{self.config.title} - {frame_data['metadata']}")
                
            # Force draw
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            self.logger.error(f"Visualization error: {e}")
            
    def start_animation(self, data_generator: Callable):
        """Start real-time animation"""
        try:
            self.animation = FuncAnimation(
                self.fig,
                self.update_plot,
                data_generator,
                interval=self.config.update_interval,
                blit=False
            )
            plt.show()
        except Exception as e:
            self.logger.error(f"Animation error: {e}")
            
    def stop_animation(self):
        """Stop the animation"""
        if self.animation:
            self.animation.event_source.stop()
            
    def save_frame(self, filename: str):
        """Save current frame to file"""
        try:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"Frame saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")
