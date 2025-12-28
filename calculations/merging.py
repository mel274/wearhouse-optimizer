"""
Route merging and consolidation logic.
"""
import logging
from typing import List
from .types import Cluster, MatrixData, OptimizationParams
from . import metrics

logger = logging.getLogger(__name__)

def merge_routes(
    routes_data: List[Cluster], 
    matrix_data: MatrixData, 
    demands: List[int],
    params: OptimizationParams
) -> List[Cluster]:
    """
    Merges small routes into larger ones if capacity and time allow.
    """
    routes = [r['indices'][:] for r in routes_data]
    distances = matrix_data['distances']
    durations = matrix_data['durations']
    capacity = params['capacity']
    max_shift = params['max_shift_seconds']
    service_time = params['service_time_seconds']
    
    improved = True
    while improved:
        improved = False
        for i in range(len(routes)):
            if not routes[i]: continue
            
            for j in range(len(routes)):
                if i == j or not routes[j]: continue
                
                # Check Capacity
                combined_load = sum(demands[x] for x in routes[i] + routes[j])
                if combined_load > capacity:
                    continue

                # Check Time
                combined_route = routes[i] + routes[j]
                if metrics.calc_approx_route_duration(combined_route, durations, service_time) > max_shift:
                    continue

                # Check Distance Savings (Stem mileage reduction)
                d_i = metrics.calc_approx_route_dist(routes[i], distances)
                d_j = metrics.calc_approx_route_dist(routes[j], distances)
                d_comb = metrics.calc_approx_route_dist(combined_route, distances)
                
                # Merge if the combined distance is less than sum of individual distances
                if d_comb < (d_i + d_j):
                    routes[i] = combined_route
                    routes[j] = []
                    improved = True
                    break
            if improved: break
    
    return [{'indices': r} for r in routes if r]