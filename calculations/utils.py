"""
Utility functions for calculation modules.
"""
import math
from typing import List, Tuple

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