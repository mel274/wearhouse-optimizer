"""
Custom exceptions for Warehouse Location & Route Optimizer.
"""

class WarehouseOptimizerError(Exception):
    """Base exception for the application."""
    pass

class ConfigurationError(WarehouseOptimizerError):
    """Raised when configuration is invalid or missing."""
    pass

class GeocodingError(WarehouseOptimizerError):
    """Raised when geocoding fails for all services."""
    pass

class APIRateLimitError(WarehouseOptimizerError):
    """Raised when API rate limits are exceeded."""
    pass

class DataValidationError(WarehouseOptimizerError):
    """Raised when input data validation fails."""
    pass

class OptimizationError(WarehouseOptimizerError):
    """Raised when route optimization fails."""
    pass

class CacheError(WarehouseOptimizerError):
    """Raised when cache operations fail."""
    pass
