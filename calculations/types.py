"""
Shared type definitions for the warehouse optimization system.
"""
from typing import List, Dict, Any, TypedDict, Optional, Tuple, Union

# Type alias for coordinates (lat, lng)
Coords = Tuple[float, float]

class OptimizationParams(TypedDict):
    fleet_size: int
    capacity: int
    max_shift_seconds: int
    service_time_seconds: int

class MatrixData(TypedDict):
    durations: List[List[float]]
    distances: List[List[float]]

class RouteMetrics(TypedDict):
    distance: float
    duration: float
    polylines: List[List[Tuple[float, float]]]

class UnservedCustomer(TypedDict):
    id: int  # Original customer index
    reason: str
    demand: int

class Solution(TypedDict):
    solution_found: bool
    routes: List[List[int]]
    route_metrics: List[RouteMetrics]
    unserved: List[UnservedCustomer]  # New field for failed customers
    metrics: Dict[str, Any]

class Cluster(TypedDict):
    indices: List[int]