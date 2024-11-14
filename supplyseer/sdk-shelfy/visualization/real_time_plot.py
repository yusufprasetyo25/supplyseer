"""
visualization/real_time_plot.py
------------------------------
Multi-plot real-time visualization.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import Optional, Callable, Dict, List
import logging
from dataclasses import dataclass
import time
from collections import deque

@dataclass
class PlotConfig:
    enabled: bool
    update_interval: int
    window_size: tuple
    history_length: int = 100
    plot_types: List[str] = None

class RealTimePlotter:
    """Real-time multi-plot visualization"""
    
    def __init__(self, config: PlotConfig):
        self.config = config
        self.fig = None
        self.axes = {}
        self.plots = {}
        self.history = {}
        self.animation = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize history for each plot type
        self._init_history()
        
    def _init_history(self):
        """Initialize data history for each plot type"""
        plot_types = self.config.plot_types or ['heatmap', 'histogram', 'timeline', 'cop']
        for plot_type in plot_types:
            self.history[plot_type] = deque(maxlen=self.config.history_length)
            
    def setup_plots(self):
        """Initialize all plot components"""
        # Create figure with subplots
        self.fig, axes = plt.subplots(2, 2, figsize=self.config.window_size)
        self.axes = {
            'heatmap': axes[0, 0],
            'histogram': axes[0, 1],
            'timeline': axes[1, 0],
            'cop': axes[1, 1]
        }
        
        # Setup individual plots
        self._setup_heatmap()
        self._setup_histogram()
        self._setup_timeline()
        self._setup_cop()
        
        plt.tight_layout()
        
    def _setup_heatmap(self):
        """Setup pressure heatmap"""
        self.plots['heatmap'] = self.axes['heatmap'].imshow(
            np.zeros((15, 15)),
            cmap='YlOrRd',
            interpolation='nearest'
        )
        self.axes['heatmap'].set_title('Pressure Distribution')
        plt.colorbar(self.plots['heatmap'], ax=self.axes['heatmap'])
        
    def _setup_histogram(self):
        """Setup pressure histogram"""
        self.plots['histogram'], = self.axes['histogram'].hist(
            [], bins=50, range=(0, 1)
        )
        self.axes['histogram'].set_title('Pressure Histogram')
        self.axes['histogram'].set_xlabel('Pressure')
        self.axes['histogram'].set_ylabel('Count')
        
    def _setup_timeline(self):
        """Setup pressure timeline"""
        self.plots['timeline'], = self.axes['timeline'].plot([], [])
        self.axes['timeline'].set_title('Pressure Over Time')
        self.axes['timeline'].set_xlabel('Time (s)')
        self.axes['timeline'].set_ylabel('Mean Pressure')
        self.axes['timeline'].grid(True)
        
    def _setup_cop(self):
        """Setup center of pressure plot"""
        self.plots['cop'] = self.axes['cop'].scatter([], [])
        self.axes['cop'].set_title('Center of Pressure')
        self.axes['cop'].set_xlabel('X Position')
        self.axes['cop'].set_ylabel('Y Position')
        self.axes['cop'].set_xlim(0, 14)
        self.axes['cop'].set_ylim(0, 14)
        self.axes['cop'].grid(True)
        
    def update_plots(self, frame_data: Dict):
        """Update all plots with new data"""
        try:
            # Update history
            current_time = time.time()
            for plot_type in self.history:
                if plot_type in frame_data:
                    self.history[plot_type].append((current_time, frame_data[plot_type]))
                    
            # Update each plot
            self._update_heatmap(frame_data)
            self._update_histogram(frame_data)
            self._update_timeline()
            self._update_cop(frame_data)
            
        except Exception as e:
            self.logger.error(f"Plot update error: {e}")
            
    def _update_heatmap(self, frame_data: Dict):
        """Update pressure heatmap"""
        if 'pressure' in frame_data:
            self.plots['heatmap'].set_array(frame_data['pressure'])
            
    def _update_histogram(self, frame_data: Dict):
        """Update pressure histogram"""
        if 'pressure' in frame_data:
            self.axes['histogram'].clear()
            self.axes['histogram'].hist(
                frame_data['pressure'].flatten(),
                bins=50,
                range=(0, 1)
            )
            self.axes['histogram'].set_title('Pressure Histogram')
            
    def _update_timeline(self):
        """Update pressure timeline"""
        if self.history['pressure']:
            times, pressures = zip(*[
                (t - self.history['pressure'][0][0], np.mean(p))
                for t, p in self.history['pressure']
            ])
            self.plots['timeline'].set_data(times, pressures)
            self.axes['timeline'].relim()
            self.axes['timeline'].autoscale_view()
            
    def _update_cop(self, frame_data: Dict):
        """Update center of pressure plot"""
        if 'features' in frame_data and 'center_of_pressure' in frame_data['features']:
            cop = frame_data['features']['center_of_pressure']
            self.plots['cop'].set_offsets([cop])
            
    def start_animation(self, data_generator: Callable):
        """Start real-time animation"""
        if not self.fig:
            self.setup_plots()
            
        try:
            self.animation = FuncAnimation(
                self.fig,
                self.update_plots,
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
            
    def save_plots(self, filename: str):
        """Save current plots to file"""
        try:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plots saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving plots: {e}")