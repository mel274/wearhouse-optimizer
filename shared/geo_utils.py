"""
Shared geographic utilities for consistent behavior across the codebase.
"""
from typing import List, Tuple, Set
from config import Config


def identify_gush_dan_customers(coords: List[Tuple[float, float]]) -> Set[int]:
    """
    Identify customer nodes within Gush Dan bounds.

    IMPORTANT: This is the SINGLE SOURCE OF TRUTH for Gush Dan detection.
    Used by both RouteOptimizer and ClusterService to avoid drift.

    Args:
        coords: List of (lat, lng) coordinates where index 0 is depot

    Returns:
        Set of matrix indices (1-based) that are in Gush Dan
    """
    gush_dan_indices = set()
    bounds = Config.GUSH_DAN_BOUNDS

    # Skip index 0 (depot), check customers starting at index 1
    for node_idx in range(1, len(coords)):
        lat, lng = coords[node_idx]
        if (bounds['min_lat'] <= lat <= bounds['max_lat'] and
            bounds['min_lon'] <= lng <= bounds['max_lon']):
            gush_dan_indices.add(node_idx)

    return gush_dan_indices
