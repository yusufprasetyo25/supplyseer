"""
visualization/data_export.py
---------------------------
Data export and storage functionality.
"""

import pandas as pd
import numpy as np
import json
import h5py
from pathlib import Path
from typing import Dict, List, Union, Any
import logging
from datetime import datetime
import csv
from dataclasses import dataclass

@dataclass
class ExportConfig:
    format: str
    compression: bool
    auto_export: bool
    export_interval: int
    directories: Dict[str, str]

class DataExporter:
    """Handles data export in various formats"""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_directories()
        self.current_session = None
        self.session_data = []
        
    def _setup_directories(self):
        """Create necessary directories"""
        for dir_name, dir_path in self.config.directories.items():
            path = Path(dir_path)
            path.mkdir(parents=True, exist_ok=True)
            
    def start_session(self):
        """Start a new export session"""
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_data.clear()
        self.logger.info(f"Started new export session: {self.current_session}")
        
    def export_frame(self, frame_data: Dict):
        """Export a single frame of data"""
        if self.config.auto_export:
            self.session_data.append(frame_data)
            
            # Export if interval reached
            if len(self.session_data) >= self.config.export_interval:
                self.export_session()
                self.session_data.clear()
                
    def export_session(self, format: str = None):
        """Export complete session data"""
        if not self.current_session:
            self.start_session()
            
        format = format or self.config.format
        
        try:
            if format == 'hdf5':
                self._export_hdf5()
            elif format == 'csv':
                self._export_csv()
            elif format == 'json':
                self._export_json()
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            self.logger.info(f"Session exported in {format} format")
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            raise
            
    def _export_hdf5(self):
        """Export data in HDF5 format"""
        filename = Path(self.config.directories['exports']) / f"session_{self.current_session}.h5"
        
        with h5py.File(filename, 'w') as f:
            # Create groups
            data_group = f.create_group('data')
            metadata_group = f.create_group('metadata')
            
            # Store pressure data
            pressure_data = np.array([frame['pressure'] for frame in self.session_data])
            data_group.create_dataset(
                'pressure',
                data=pressure_data,
                compression='gzip' if self.config.compression else None
            )
            
            # Store timestamps
            timestamps = [frame['timestamp'] for frame in self.session_data]
            data_group.create_dataset('timestamps', data=timestamps)
            
            # Store features
            if 'features' in self.session_data[0]:
                features_group = data_group.create_group('features')
                for feature_name in self.session_data[0]['features']:
                    feature_data = [frame['features'][feature_name] for frame in self.session_data]
                    features_group.create_dataset(feature_name, data=feature_data)
                    
            # Store metadata
            metadata_group.attrs['session_id'] = self.current_session
            metadata_group.attrs['num_frames'] = len(self.session_data)
            metadata_group.attrs['timestamp'] = datetime.now().isoformat()
            
    def _export_csv(self):
        """Export data in CSV format"""
        base_path = Path(self.config.directories['exports'])
        
        # Export pressure data
        pressure_file = base_path / f"pressure_{self.current_session}.csv"
        with open(pressure_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp'] + [f'sensor_{i}' for i in range(225)])  # 15x15 = 225
            for frame in self.session_data:
                row = [frame['timestamp']] + frame['pressure'].flatten().tolist()
                writer.writerow(row)
                
        # Export features if available
        if 'features' in self.session_data[0]:
            features_file = base_path / f"features_{self.current_session}.csv"
            features = self.session_data[0]['features'].keys()
            with open(features_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp'] + list(features))
                for frame in self.session_data:
                    row = [frame['timestamp']] + [
                        frame['features'][feature] for feature in features
                    ]
                    writer.writerow(row)
                    
        # Export metadata
        metadata_file = base_path / f"metadata_{self.current_session}.csv"
        with open(metadata_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['session_id', 'num_frames', 'timestamp'])
            writer.writerow([
                self.current_session,
                len(self.session_data),
                datetime.now().isoformat()
            ])
            
    def _export_json(self):
        """Export data in JSON format"""
        filename = Path(self.config.directories['exports']) / f"session_{self.current_session}.json"
        
        # Prepare data for JSON serialization
        json_data = {
            'metadata': {
                'session_id': self.current_session,
                'num_frames': len(self.session_data),
                'timestamp': datetime.now().isoformat()
            },
            'frames': []
        }
        
        # Convert frame data to JSON-serializable format
        for frame in self.session_data:
            json_frame = {
                'timestamp': frame['timestamp'],
                'pressure': frame['pressure'].tolist()
            }
            
            if 'features' in frame:
                json_frame['features'] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in frame['features'].items()
                }
                
            json_data['frames'].append(json_frame)
            
        # Write to file
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
            
    def load_session(self, session_id: str, format: str = None):
        """Load a previously exported session"""
        format = format or self.config.format
        base_path = Path(self.config.directories['exports'])
        
        try:
            if format == 'hdf5':
                return self._load_hdf5(base_path / f"session_{session_id}.h5")
            elif format == 'csv':
                return self._load_csv(base_path, session_id)
            elif format == 'json':
                return self._load_json(base_path / f"session_{session_id}.json")
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            self.logger.error(f"Error loading session: {e}")
            raise
            
    def _load_hdf5(self, filepath: Path) -> Dict:
        """Load session data from HDF5 file"""
        with h5py.File(filepath, 'r') as f:
            data = {
                'pressure': np.array(f['data/pressure']),
                'timestamps': np.array(f['data/timestamps'])
            }
            
            # Load features if available
            if 'features' in f['data']:
                data['features'] = {
                    name: np.array(f[f'data/features/{name}'])
                    for name in f['data/features']
                }
                
            # Load metadata
            data['metadata'] = dict(f['metadata'].attrs)
            
        return data
        
    def _load_csv(self, base_path: Path, session_id: str) -> Dict:
        """Load session data from CSV files"""
        # Load pressure data
        pressure_df = pd.read_csv(base_path / f"pressure_{session_id}.csv")
        timestamps = pressure_df['timestamp'].values
        pressure_data = pressure_df.drop('timestamp', axis=1).values
        
        data = {
            'timestamps': timestamps,
            'pressure': pressure_data.reshape(-1, 15, 15)
        }
        
        # Load features if available
        features_file = base_path / f"features_{session_id}.csv"
        if features_file.exists():
            features_df = pd.read_csv(features_file)
            data['features'] = features_df.drop('timestamp', axis=1).to_dict('list')
            
        # Load metadata
        metadata_file = base_path / f"metadata_{session_id}.csv"
        if metadata_file.exists():
            metadata_df = pd.read_csv(metadata_file)
            data['metadata'] = metadata_df.iloc[0].to_dict()
            
        return data
        
    def _load_json(self, filepath: Path) -> Dict:
        """Load session data from JSON file"""
        with open(filepath) as f:
            json_data = json.load(f)
            
        # Convert data back to numpy arrays
        frames = json_data['frames']
        data = {
            'metadata': json_data['metadata'],
            'timestamps': np.array([frame['timestamp'] for frame in frames]),
            'pressure': np.array([frame['pressure'] for frame in frames])
        }
        
        # Convert features if available
        if 'features' in frames[0]:
            data['features'] = {
                feature: np.array([frame['features'][feature] for frame in frames])
                for feature in frames[0]['features']
            }
            
        return data