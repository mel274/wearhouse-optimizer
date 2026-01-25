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

class UnroutablePointError(WarehouseOptimizerError):
    """Raised when a coordinate cannot be routed by the ORS API."""
    def __init__(self, index: int, message: str = ""):
        self.index = index
        self.message = message
        super().__init__(f"Coordinate at index {index} is unroutable: {message}")
