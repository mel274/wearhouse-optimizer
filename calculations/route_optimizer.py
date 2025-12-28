"""
Main orchestration for the Warehouse Route Optimization.
Replaces the monolithic solver.py.
"""
import logging
import time
from typing import List, Dict, Any, Tuple
from .types import Solution, OptimizationParams
from . import k_cluster, prioritization, merging, ors, metrics, utils

logger = logging.getLogger(__name__)

class RouteOptimizer:
    """
    Orchestrates the optimization pipeline:
    1. Fetch Matrix -> 2. Cluster -> 3. Prioritize -> 4. Merge -> 5. Finalize (ORS)
    """
    
    def __init__(self, api_key: str):
        self.ors_handler = ors.ORSHandler(api_key)

    def solve(self, 
              locations: List[str], 
              demands: List[int], 
              params: OptimizationParams, 
              coords: List[Tuple[float, float]]) -> Solution:
        """
        Executes the optimization pipeline.
        """
        logger.info(f"Starting Optimization for {len(locations)} locations.")
        
        try:
            # --- Phase 0: Data Prep & Matrix ---
            if not coords:
                raise ValueError("No coordinates provided.")
            
            logger.info("Phase 0: Fetching Distance Matrix...")
            matrix_data = self.ors_handler.get_distance_matrix(coords)
            time.sleep(1) # Gentle cool-down

            customer_coords = coords[1:]
            customer_demands = demands[1:]
            
            # --- Phase 1: K-Means Clustering ---
            logger.info("Phase 1: Initial Clustering...")
            initial_clusters = k_cluster.perform_clustering(
                customer_coords, customer_demands, 
                params['fleet_size'], params['capacity']
            )

            # --- Phase 2: Global Relocation (Corridor Scavenging) ---
            logger.info("Phase 2: Global Relocation...")
            refined_clusters = prioritization.perform_global_relocation(
                initial_clusters, matrix_data, demands, params
            )

            # --- Phase 3: Merging ---
            logger.info("Phase 3: Route Merging...")
            merged_clusters = merging.merge_routes(
                refined_clusters, matrix_data, demands, params
            )

            # --- Phase 4: Final Optimization (Per Truck) ---
            logger.info("Phase 4: Final Route Optimization...")
            final_solution = self._finalize_routes(
                merged_clusters, coords, demands, params
            )

            return final_solution

        except Exception as e:
            logger.error(f"Optimization Failed: {e}")
            raise

    def _finalize_routes(self, 
                         clusters: List[Dict], 
                         coords: List[Tuple[float, float]], 
                         demands: List[int],
                         params: OptimizationParams) -> Solution:
        """
        Sends each cluster to ORS for detailed path optimization.
        """
        final_routes = []
        route_metrics = []
        warehouse_coord = coords[0]

        for i, cluster in enumerate(clusters):
            indices = cluster['indices']
            if not indices: continue
            
            # Construct ORS VRP Request
            # Note: We treat each truck as an independent VRP problem for ORS
            # to ensure it strictly follows our clustering logic.
            vehicle = {
                "id": 1, "profile": "driving-car",
                "start": [warehouse_coord[1], warehouse_coord[0]],
                "end": [warehouse_coord[1], warehouse_coord[0]],
                "capacity": [params['capacity']], 
                "time_window": [0, params['max_shift_seconds']]
            }
            jobs = []
            for cust_idx in indices:
                c_coords = coords[cust_idx]
                jobs.append({
                    "id": cust_idx, 
                    "service": params['service_time_seconds'],
                    "amount": [demands[cust_idx]], 
                    "location": [c_coords[1], c_coords[0]]
                })
            
            request_body = {"vehicles": [vehicle], "jobs": jobs, "options": {"g": True}}
            
            try:
                # Rate limit sleep
                time.sleep(0.3)
                api_res = self.ors_handler.optimize_route(request_body)
                
                if 'routes' in api_res and api_res['routes']:
                    route_data = api_res['routes'][0]
                    
                    # Extract sequence (0 -> customers -> 0)
                    steps = [step['job'] for step in route_data.get('steps', []) if step['type'] == 'job']
                    full_route = [0] + steps + [0]
                    
                    polylines = self.ors_handler.get_route_polylines(route_data, warehouse_coord)
                    
                    final_routes.append(full_route)
                    route_metrics.append({
                        'distance': route_data.get('distance', 0),
                        'duration': route_data.get('duration', 0) + (len(steps) * params['service_time_seconds']),
                        'polylines': polylines
                    })
                else:
                    logger.warning(f"Truck {i}: ORS returned no route. Using fallback.")
                    self._apply_fallback(indices, final_routes, route_metrics, coords, matrix_data=None)

            except Exception as e:
                logger.error(f"Truck {i} Optimization Failed: {e}. Using fallback.")
                self._apply_fallback(indices, final_routes, route_metrics, coords, matrix_data=None)

        # Aggregate totals
        total_dist = sum(r['distance'] for r in route_metrics)
        total_time = sum(r['duration'] for r in route_metrics)

        return {
            'solution_found': True,
            'routes': final_routes,
            'route_metrics': route_metrics,
            'metrics': {
                'total_distance': total_dist,
                'total_time': total_time,
                'num_vehicles_used': len(final_routes)
            }
        }

    def _apply_fallback(self, indices, routes_list, metrics_list, coords, matrix_data):
        """Simple Geometric sort fallback."""
        import math
        def get_dist(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        curr = 0 # Warehouse index
        unvisited = set(indices)
        path = []
        while unvisited:
            nxt = min(unvisited, key=lambda x: get_dist(coords[curr], coords[x]))
            unvisited.remove(nxt)
            path.append(nxt)
            curr = nxt
        
        full_route = [0] + path + [0]
        routes_list.append(full_route)
        metrics_list.append({'distance': 0, 'duration': 0, 'polylines': []})