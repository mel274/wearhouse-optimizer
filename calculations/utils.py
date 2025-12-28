"""
Utility functions for calculation modules.
"""
import time
import logging
import math
from typing import Callable, List, Tuple
from functools import wraps

logger = logging.getLogger(__name__)

def retry_with_backoff(max_attempts: int = 3, delay: float = 2.0):
    """Decorator for retrying operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait_time = delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            raise last_exception
        return wrapper
    return decorator

def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    """Decodes a Google-encoded polyline string into (lat, lng) tuples."""
    index, lat, lng = 0, 0, 0
    coordinates = []
    length = len(polyline_str)
    
    try:
        while index < length:
            shift, result = 0, 0
            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if byte < 0x20: break
            dlat = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += dlat

            shift, result = 0, 0
            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if byte < 0x20: break
            dlng = ~(result >> 1) if (result & 1) else (result >> 1)
            lng += dlng

            coordinates.append((lat / 1e5, lng / 1e5))
        return coordinates
    except Exception as e:
        logger.error(f"Error decoding polyline: {e}")
        return []