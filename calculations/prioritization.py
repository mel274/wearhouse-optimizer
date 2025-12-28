"""
Route prioritization and global relocation logic (Scavenging).
"""
import logging
from typing import List, Dict, Any
from .types import Cluster, MatrixData, OptimizationParams
from . import metrics

logger = logging.getLogger(__name__)

def perform_global_relocation(
    routes_data: List[Cluster], 
    matrix_data: MatrixData, 
    demands: List[int],
    params: OptimizationParams
) -> List[Cluster]:
    """
    Moves customers between routes to minimize global cost.
    Prioritizes filling 'Long Haul' corridors.
    """
    routes = [r['indices'][:] for r in routes_data]
    distances = matrix_data['distances']
    durations = matrix_data['durations']
    capacity = params['capacity']
    max_shift = params['max_shift_seconds']
    service_time = params['service_time_seconds']

    improved = True
    iteration = 0
    max_iterations = 50

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # 1. Sort Routes by "Reach" (furthest customer from warehouse)
        route_priorities = []
        for r_idx, route in enumerate(routes):
            if not route:
                route_priorities.append((-1, r_idx))
                continue
            max_dist = max([metrics.get_matrix_val(distances, 0, n) for n in route])
            route_priorities.append((max_dist, r_idx))
        
        # Sort descending: Process longest routes (corridors) first
        route_priorities.sort(key=lambda x: x[0], reverse=True)
        sorted_indices = [x[1] for x in route_priorities]

        # 2. Scavenging Loop
        for target_idx in sorted_indices:
            target_route = routes[target_idx]
            if not target_route: continue
            
            current_load = sum(demands[i] for i in target_route)
            
            for source_idx in range(len(routes)):
                if target_idx == source_idx: continue
                source_route = routes[source_idx]
                if not source_route: continue
                
                # Try moving customers from Source to Target
                for cust_idx in source_route[:]:
                    cust_demand = demands[cust_idx]
                    
                    # Capacity Check
                    if current_load + cust_demand > capacity:
                        continue
                        
                    # Cost Analysis
                    savings_source = metrics.calc_removal_savings(cust_idx, source_route, distances)
                    cost_target = metrics.calc_insertion_cost(cust_idx, target_route, distances)
                    net_impact = cost_target - savings_source
                    
                    # Heuristic: Accept if saves distance or adds very little (< 100m)
                    # This helps build density in the target corridor
                    if net_impact < 100:
                        # Time Check (Expensive, so do last)
                        simulated_target = target_route + [cust_idx]
                        new_duration = metrics.calc_approx_route_duration(simulated_target, durations, service_time)
                        
                        if new_duration <= max_shift:
                            # Apply Move
                            source_route.remove(cust_idx)
                            target_route.append(cust_idx)
                            current_load += cust_demand
                            improved = True
                            break # Restart search on change
                
                if improved: break
            if improved: break
            
    return [{'indices': r} for r in routes if r]