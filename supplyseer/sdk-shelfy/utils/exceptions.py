"""
utils/exceptions.py
------------------
Custom exceptions for the system.
"""

class VelostatError(Exception):
    """Base exception for Velostat system"""
    pass

class SensorError(VelostatError):
    """Sensor-related errors"""
    pass

class CalibrationError(VelostatError):
    """Calibration-related errors"""
    pass

class ProcessingError(VelostatError):
    """Data processing errors"""
    pass

class ConfigError(VelostatError):
    """Configuration-related errors"""
    pass

class ConnectionError(VelostatError):
    """Connection-related errors"""
    pass

class ExportError(VelostatError):
    """Data export errors"""
    pass
