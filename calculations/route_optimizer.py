"""
Main orchestration for the Warehouse Route Optimization.
Version: 2.1 (Enhanced Failure Handling)
"""
import logging
import time
from typing import List, Dict, Any, Tuple, Set
from .types import Solution, OptimizationParams, UnservedCustomer
from . import k_cluster, prioritization, merging, ors, metrics, utils

logger = logging.getLogger(__name__)

class RouteOptimizer:
    """
    Orchestrates the optimization pipeline:
    1. Fetch Matrix -> 2. Cluster -> 3. Prioritize -> 4. Merge -> 5. Finalize (ORS)
    Includes logic to track customers that fall out of the solution.
    """
    
    def __init__(self, api_key: str):
        self.ors_handler = ors.ORSHandler(api_key)

    def _run_optimization_pipeline(self, locations, demands, coords, customer_coords, 
                                 customer_demands, params, matrix_data):
        """Runs the optimization pipeline with the given parameters."""
        all_customer_indices = set(range(1, len(locations)))
        
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
        return self._finalize_routes(
            merged_clusters, coords, demands, params, all_customer_indices
        )

    def solve(self, 
              locations: List[str], 
              demands: List[int], 
              params: OptimizationParams, 
              coords: List[Tuple[float, float]]) -> Solution:
        """
        Executes the optimization pipeline with support for handling unserved customers.
        """
        logger.info(f"Starting Optimization for {len(locations)} locations.")
        
        try:
            # --- Phase 0: Data Prep & Matrix ---
            if not coords:
                raise ValueError("No coordinates provided.")
            
            logger.info("Phase 0: Fetching Distance Matrix...")
            matrix_data = self.ors_handler.get_distance_matrix(coords)
            time.sleep(1)  # Gentle cool-down

            customer_coords = coords[1:]
            customer_demands = demands[1:]
            
            # Initialize tracking of all customer indices (1-based)
            all_customer_indices = set(range(1, len(locations)))
            
            # Run the optimization pipeline
            solution = self._run_optimization_pipeline(
                locations, demands, coords, 
                customer_coords, customer_demands,
                params, matrix_data
            )
            
            return solution

        except Exception as e:
            logger.error(f"Optimization Failed: {e}")
            raise

    def _create_new_route(self, customer_indices: List[int], demands: List[int], capacity: int) -> List[List[int]]:
        """Create new routes from unserved customers, respecting vehicle capacity."""
        routes = []
        current_route = []
        current_load = 0
        
        for idx in sorted(customer_indices, key=lambda x: -demands[x]):  # Sort by demand descending
            if current_load + demands[idx] <= capacity:
                current_route.append(idx)
                current_load += demands[idx]
            else:
                if current_route:  # Only add non-empty routes
                    routes.append(current_route)
                current_route = [idx]
                current_load = demands[idx]
        
        if current_route:  # Add the last route if not empty
            routes.append(current_route)
            
        return routes

    def _group_unserved_by_proximity(self, unserved_indices, coords, demands, capacity, warehouse_coord):
        """Group unserved customers by geographical proximity, considering warehouse location."""
        if not unserved_indices:
            return []
        
        # Calculate distances from warehouse
        def distance_to_warehouse(idx):
            x, y = coords[idx]
            wx, wy = warehouse_coord
            return ((x - wx)**2 + (y - wy)**2)**0.5
            
        # Sort customers by distance from warehouse (closest first)
        customers_by_warehouse_dist = sorted(unserved_indices, key=distance_to_warehouse)
        
        routes = []
        remaining_customers = set(customers_by_warehouse_dist)
        
        # Use max_vehicles from self if available, otherwise use a reasonable default or pass it in.
        # Assuming self.max_vehicles is set or we can use a passed parameter.
        max_new_routes = getattr(self, 'max_vehicles', len(unserved_indices)) 

        while remaining_customers and len(routes) < max_new_routes:
            # Start with the customer closest to the warehouse
            current_customer = min(remaining_customers, key=distance_to_warehouse)
            current_route = [current_customer]
            current_load = demands[current_customer]
            remaining_customers.remove(current_customer)
            
            # Find nearest neighbors to the current route
            while remaining_customers and current_load < capacity:
                # Calculate distance from each remaining customer to the current route
                def min_route_distance(cust_idx):
                    min_dist = float('inf')
                    for route_cust in current_route:
                        x1, y1 = coords[cust_idx]
                        x2, y2 = coords[route_cust]
                        dist = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                        min_dist = min(min_dist, dist)
                    return min_dist
                    
                # Find the closest customer that fits in capacity
                next_customer = None
                min_dist = float('inf')
                
                for cust in remaining_customers:
                    if demands[cust] + current_load <= capacity:
                        dist = min_route_distance(cust)
                        if dist < min_dist:
                            min_dist = dist
                            next_customer = cust
                
                if next_customer is None:
                    break  # No more customers can fit
                    
                current_route.append(next_customer)
                current_load += demands[next_customer]
                remaining_customers.remove(next_customer)
                
            if current_route:
                routes.append(current_route)
                
        return routes

    def _finalize_routes(self, 
                         clusters: List[Dict], 
                         coords: List[Tuple[float, float]], 
                         demands: List[int],
                         params: OptimizationParams,
                         all_indices: Set[int]) -> Solution:
        """
        Sends each cluster to ORS for detailed path optimization.
        Handles unserved customers by creating new routes when possible.
        Returns a solution with detailed metrics and optimization suggestions.
        """
        final_routes = []
        route_metrics = []
        unserved_customers: List[UnservedCustomer] = []
        served_indices = set()
        
        warehouse_coord = coords[0]
        matrix_data = self.ors_handler.get_distance_matrix(coords)
        self.max_vehicles = params['fleet_size']
        max_vehicles = self.max_vehicles

        for i, cluster in enumerate(clusters):
            indices = cluster['indices']
            if not indices: 
                continue
                
            # Initial setup for this cluster
            current_indices = indices.copy()
            valid_indices_for_api = []
            
            # Check for single-item overloads first
            for cust_idx in indices:
                if demands[cust_idx] > params['capacity']:
                    unserved_customers.append({
                        'id': cust_idx,
                        'reason': 'Demand exceeds vehicle capacity',
                        'demand': demands[cust_idx]
                    })
                    logger.warning(f"Customer {cust_idx} dropped: Demand {demands[cust_idx]} > Capacity {params['capacity']}")
                else:
                    valid_indices_for_api.append(cust_idx)
            
            if not valid_indices_for_api:
                continue
                
            # Sort customers by distance from warehouse (furthest first) for removal priority
            valid_indices_for_api.sort(
                key=lambda x: matrix_data['distances'][0][x], 
                reverse=True
            )
            
            # Main optimization loop with graceful degradation
            route_found = False
            while valid_indices_for_api and not route_found:
                # Prepare jobs for current set of customers
                jobs = []
                for cust_idx in valid_indices_for_api:
                    c_coords = coords[cust_idx]
                    jobs.append({
                        "id": cust_idx,
                        "service": params['service_time_seconds'],
                        "amount": [demands[cust_idx]],
                        "location": [c_coords[1], c_coords[0]]
                    })
                
                # Vehicle setup
                vehicle = {
                    "id": 1, "profile": "driving-car",
                    "start": [warehouse_coord[1], warehouse_coord[0]],
                    "end": [warehouse_coord[1], warehouse_coord[0]],
                    "capacity": [params['capacity']],
                    "time_window": [0, params['max_shift_seconds']]
                }
                
                request_body = {"vehicles": [vehicle], "jobs": jobs, "options": {"g": True}}
            
                try:
                    # Rate limit sleep
                    time.sleep(0.3)
                    api_res = self.ors_handler.optimize_route(request_body)
                    
                    # 1. Check for API/Solver Errors (Technical Failures)
                    if 'code' in api_res and api_res['code'] != 0:
                         msg = api_res.get('message', 'Unknown Error')
                         raise Exception(f"ORS Error Code {api_res['code']}: {msg}")
                    
                    current_unserved = []
                    unassigned_ids = set()
                    
                    # 2. Process Unassigned Jobs (Constraint Violations)
                    if 'unassigned' in api_res:
                        for job in api_res['unassigned']:
                            uid = job['id']
                            unassigned_ids.add(uid)
                            
                            reason = job.get('description', 'Optimization Constraint (Time Window/Routing)')
                            current_unserved.append({
                                'id': uid,
                                'reason': reason,
                                'demand': demands[uid]
                            })
                    
                    # 3. Process Routes if found
                    if 'routes' in api_res and api_res['routes']:
                        route_data = api_res['routes'][0]
                        route_duration = route_data.get('duration', 0) + (len(valid_indices_for_api) * params['service_time_seconds'])
                        
                        # Check if the route duration is within limits
                        if route_duration <= params['max_shift_seconds']:
                            # Valid route found within constraints
                            steps = [step['job'] for step in route_data.get('steps', []) if step['type'] == 'job']
                            full_route = [0] + steps + [0]
                            
                            # Track served indices
                            current_served = set(steps)
                            served_indices.update(current_served)
                            
                            # Identify any dropped customers
                            dropped_unexpectedly = set(valid_indices_for_api) - current_served - unassigned_ids
                            for dropped_idx in dropped_unexpectedly:
                                current_unserved.append({
                                    'id': dropped_idx,
                                    'reason': 'Implicitly Dropped by Optimizer',
                                    'demand': demands[dropped_idx]
                                })
                            
                            polylines = self.ors_handler.get_route_polylines(route_data, warehouse_coord)
                            
                            final_routes.append(full_route)
                            route_metrics.append({
                                'distance': route_data.get('distance', 0),
                                'duration': route_duration,
                                'polylines': polylines
                            })
                            
                            # Add any unserved customers to the global list
                            unserved_customers.extend(current_unserved)
                            route_found = True  # Exit the while loop
                            continue  # Move to next cluster
                        
                        else:
                            # Route found but exceeds time constraints
                            logger.warning(f"Truck {i}: Route duration {route_duration}s exceeds max shift {params['max_shift_seconds']}s")
                    
                    # If we get here, either no route was found or it violated constraints
                    # Remove the most expensive customer (furthest from warehouse) and try again
                    if valid_indices_for_api:
                        removed_cust = valid_indices_for_api.pop(0)  # Already sorted by distance
                        unserved_customers.append({
                            'id': removed_cust,
                            'reason': 'Removed to save route feasibility',
                            'demand': demands[removed_cust]
                        })
                        logger.info(f"Removed customer {removed_cust} to improve route feasibility")
                    
                except Exception as e:
                    # Handle API errors - remove a customer and retry
                    logger.error(f"Truck {i} Optimization Error: {e}")
                    if valid_indices_for_api:
                        removed_cust = valid_indices_for_api.pop(0)  # Already sorted by distance
                        unserved_customers.append({
                            'id': removed_cust,
                            'reason': f'Removed due to optimization error: {str(e)}',
                            'demand': demands[removed_cust]
                        })
                        logger.info(f"Removed customer {removed_cust} after optimization error")
            
            # If we exited the loop without finding a valid route, apply fallback to remaining customers
            if valid_indices_for_api and not route_found:
                logger.warning(f"Truck {i}: Could not find valid route, applying fallback to {len(valid_indices_for_api)} customers")
                self._apply_fallback(valid_indices_for_api, final_routes, route_metrics, coords, served_indices)

        # Handle unserved customers by creating new routes if we have available vehicles
        remaining_unserved = all_indices - served_indices - {u['id'] for u in unserved_customers}
        
        # Add any unaccounted customers to unserved list
        for idx in remaining_unserved:
            unserved_customers.append({
                'id': idx,
                'reason': 'Not assigned to any route',
                'demand': demands[idx]
            })
        
        # Try to create new routes for unserved customers if we have vehicle capacity
        if unserved_customers and len(final_routes) < max_vehicles:
            unserved_indices = [u['id'] for u in unserved_customers]
            remaining_vehicles = max_vehicles - len(final_routes)
            
            # Group unserved customers by proximity to warehouse and each other
            new_routes = self._group_unserved_by_proximity(
                unserved_indices=unserved_indices,
                coords=coords,
                demands=demands,
                capacity=params['capacity'],
                warehouse_coord=warehouse_coord
            )
            
            # Limit to remaining vehicle capacity
            new_routes = new_routes[:remaining_vehicles]
            
            if new_routes:
                logger.info(f"Attempting to create and validate {len(new_routes)} new routes for unserved customers.")
                newly_served_indices = set()

                for route_idx, route in enumerate(new_routes):
                    if not route:
                        continue

                    # Prepare jobs for the new route
                    jobs = []
                    for cust_idx in route:
                        c_coords = coords[cust_idx]
                        jobs.append({
                            "id": cust_idx,
                            "service": params['service_time_seconds'],
                            "amount": [demands[cust_idx]],
                            "location": [c_coords[1], c_coords[0]]
                        })

                    # Vehicle setup for the new route
                    vehicle = {
                        "id": len(final_routes) + 1, "profile": "driving-car",
                        "start": [warehouse_coord[1], warehouse_coord[0]],
                        "end": [warehouse_coord[1], warehouse_coord[0]],
                        "capacity": [params['capacity']],
                        "time_window": [0, params['max_shift_seconds']]
                    }

                    request_body = {"vehicles": [vehicle], "jobs": jobs, "options": {"g": True}}

                    try:
                        time.sleep(0.3) # Rate limit
                        api_res = self.ors_handler.optimize_route(request_body)

                        if 'routes' in api_res and api_res['routes']:
                            route_data = api_res['routes'][0]
                            route_duration = route_data.get('duration', 0) + (len(route) * params['service_time_seconds'])

                            if route_duration <= params['max_shift_seconds'] and not api_res.get('unassigned'):
                                # This new route is valid
                                steps = [step['job'] for step in route_data.get('steps', []) if step['type'] == 'job']
                                full_route = [0] + steps + [0]
                                polylines = self.ors_handler.get_route_polylines(route_data, warehouse_coord)

                                final_routes.append(full_route)
                                route_metrics.append({
                                    'distance': route_data.get('distance', 0),
                                    'duration': route_duration,
                                    'polylines': polylines
                                })
                                newly_served_indices.update(steps)
                                logger.info(f"Successfully created and validated new route serving {len(steps)} customers.")
                            else:
                                # Route is not feasible, customers remain unserved
                                logger.warning(f"Generated new route for {len(route)} customers was not feasible and was discarded.")
                        else:
                            logger.error(f"API error while validating new route: {api_res.get('message', 'Unknown error')}")

                    except Exception as e:
                        logger.error(f"Exception while validating new route: {e}")

                # Update master lists of served/unserved customers
                if newly_served_indices:
                    served_indices.update(newly_served_indices)
                    unserved_customers = [u for u in unserved_customers if u['id'] not in newly_served_indices]

        # Analyze unserved customers and generate suggestions
        suggestions = self._generate_suggestions(
            unserved_customers, 
            demands, 
            params, 
            coords, 
            len(served_indices),
            len(all_indices) if all_indices else 1
        )

        # Calculate metrics
        total_dist = sum(r['distance'] for r in route_metrics)
        total_time = sum(r['duration'] for r in route_metrics)
        max_route_time = max((r['duration'] for r in route_metrics), default=0)

        return {
            'solution_found': True,
            'routes': final_routes,
            'route_metrics': route_metrics,
            'unserved': unserved_customers,
            'suggestions': suggestions,
            'metrics': {
                'total_distance': total_dist,
                'total_time': total_time,
                'max_route_time': max_route_time,
                'num_vehicles_used': len(final_routes),
                'customers_served': len(served_indices),
                'customers_unserved': len(unserved_customers),
                'service_rate': len(served_indices) / len(all_indices) if all_indices else 0
            }
        }

    def _generate_suggestions(self, unserved_customers, demands, params, coords, served_count, total_customers):
        """Generate optimization suggestions based on unserved customers."""
        if not unserved_customers:
            return {
                'all_served': True,
                'message': 'All customers have been successfully served!',
                'suggestions': []
            }

        suggestions = []
        unserved_demands = [u['demand'] for u in unserved_customers]
        total_unserved_demand = sum(unserved_demands)
        
        # Check capacity constraints
        if any(d > params['capacity'] for d in unserved_demands):
            oversize_customers = [
                u for u in unserved_customers 
                if u['demand'] > params['capacity']
            ]
            suggestions.append({
                'type': 'capacity_issue',
                'message': f"{len(oversize_customers)} customers exceed vehicle capacity",
                'details': [{
                    'customer_id': u['id'],
                    'demand': u['demand'],
                    'vehicle_capacity': params['capacity']
                } for u in oversize_customers],
                'suggestion': 'Consider using vehicles with higher capacity or splitting orders'
            })

        # Check time constraints
        max_possible_routes = params['fleet_size']
        if len(unserved_customers) > 0 and len(unserved_customers) <= max_possible_routes:
            suggestions.append({
                'type': 'fleet_adjustment',
                'message': f'Adding {len(unserved_customers)} more routes could serve all customers',
                'suggestion': f"Increase fleet size to {len(unserved_customers) + len(final_routes) if 'final_routes' in locals() else '?'} vehicles"
            })

        # Calculate required time adjustments
        avg_service_time = params.get('service_time_seconds', 300)  # Default 5 minutes
        total_service_time = len(unserved_customers) * avg_service_time
        
        if total_service_time > 0:
            suggestions.append({
                'type': 'time_analysis',
                'message': f'Additional time needed to serve all customers: {total_service_time/3600:.1f} hours',
                'details': {
                    'current_shift_hours': params.get('max_shift_seconds', 28800)/3600,  # 8 hours default
                    'required_additional_hours': total_service_time/3600
                }
            })

        return {
            'all_served': False,
            'message': f'Served {served_count}/{total_customers} customers',
            'suggestions': suggestions,
            'unserved_summary': {
                'count': len(unserved_customers),
                'total_demand': total_unserved_demand,
                'reasons': {
                    'capacity': len([u for u in unserved_customers if 'capacity' in u.get('reason', '').lower()]),
                    'time': len([u for u in unserved_customers if 'time' in u.get('reason', '').lower()]),
                    'distance': len([u for u in unserved_customers if 'distance' in u.get('reason', '').lower()])
                }
            }
        }

    def _apply_fallback(self, indices, routes_list, metrics_list, coords, served_indices_set):
        """Geometric sort fallback with distance calculation."""
        import math
        
        def get_dist(p1, p2):
            return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        if not indices:
            return
            
        # Start from warehouse (index 0)
        path = []
        total_distance = 0
        current = 0  # warehouse index
        unvisited = set(indices)
        
        while unvisited:
            next_point = min(unvisited, key=lambda x: get_dist(coords[current], coords[x]))
            total_distance += get_dist(coords[current], coords[next_point])
            path.append(next_point)
            unvisited.remove(next_point)
            current = next_point
        
        # Return to warehouse
        total_distance += get_dist(coords[current], coords[0])
        full_route = [0] + path + [0]
        
        # Calculate estimated time (assuming 30 km/h average speed)
        avg_speed_kmh = 30
        distance_km = total_distance / 1000  # convert to km
        duration_seconds = (distance_km / avg_speed_kmh) * 3600  # convert to seconds
        
        routes_list.append(full_route)
        metrics_list.append({
            'distance': total_distance,
            'duration': duration_seconds,
            'polylines': []  # No polylines available in fallback
        })
        served_indices_set.update(indices)