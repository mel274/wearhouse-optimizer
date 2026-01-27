"""
Shared geographic utilities for consistent behavior across the codebase.
"""
import json
import os
from typing import List, Tuple, Set

# Load zone data once at module level
_ZONES_FILE = os.path.join(os.path.dirname(__file__), '..', 'assets', 'small_truck_zones.json')
_ZONES_DATA = None


def _load_zones() -> List[dict]:
    """Load zone polygons from JSON file (cached)."""
    global _ZONES_DATA
    if _ZONES_DATA is None:
        with open(_ZONES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            _ZONES_DATA = data.get('zones', [])
    return _ZONES_DATA


def _point_in_polygon(lat: float, lng: float, polygon: List[List[float]]) -> bool:
    """
    Ray-casting algorithm to check if a point is inside a polygon.
    
    Args:
        lat: Latitude of the point
        lng: Longitude of the point
        polygon: List of [lat, lng] pairs forming a closed polygon
        
    Returns:
        True if the point is inside the polygon
    """
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        lat_i, lng_i = polygon[i]
        lat_j, lng_j = polygon[j]
        
        if ((lng_i > lng) != (lng_j > lng)) and \
           (lat < (lat_j - lat_i) * (lng - lng_i) / (lng_j - lng_i) + lat_i):
            inside = not inside
        
        j = i
    
    return inside


def is_in_small_truck_zone(lat: float, lng: float) -> bool:
    """
    Check if a coordinate falls within any small truck restricted zone.
    
    Args:
        lat: Latitude
        lng: Longitude
        
    Returns:
        True if the coordinate is in a restricted zone (Ramat Gan, Tel Aviv-Yafo, Holon, Bnei Brak)
    """
    if lat is None or lng is None:
        return False
    
    zones = _load_zones()
    for zone in zones:
        polygon = zone.get('polygon', [])
        if polygon and _point_in_polygon(lat, lng, polygon):
            return True
    
    return False


def identify_small_truck_customers(coords: List[Tuple[float, float]]) -> Set[int]:
    """
    Identify customer nodes within small truck restricted zones.

    IMPORTANT: This is the SINGLE SOURCE OF TRUTH for restricted zone detection.
    Used by both RouteOptimizer and ClusterService to avoid drift.

    Args:
        coords: List of (lat, lng) coordinates where index 0 is depot

    Returns:
        Set of matrix indices (1-based) that are in restricted zones
    """
    small_truck_indices = set()

    # Skip index 0 (depot), check customers starting at index 1
    for node_idx in range(1, len(coords)):
        lat, lng = coords[node_idx]
        if is_in_small_truck_zone(lat, lng):
            small_truck_indices.add(node_idx)

    return small_truck_indices


# Backward compatibility alias
identify_gush_dan_customers = identify_small_truck_customers
