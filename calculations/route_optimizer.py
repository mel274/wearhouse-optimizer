"""
Main orchestration for the Warehouse Route Optimization.
Version: 3.0 (OR-Tools Integration)
"""
import logging
import time
import math
from typing import List, Dict, Any, Tuple, Set
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from .types import Solution, OptimizationParams, UnservedCustomer
from . import ors, metrics
import shared.utils as utils
from config import Config

logger = logging.getLogger(__name__)

class RouteOptimizer:
    """
    Orchestrates the optimization pipeline using OR-Tools:
    1. Fetch Matrix (cached) -> 2. OR-Tools Global Solver -> 3. Hydrate with ORS Directions
    """
    
    def __init__(self, api_key: str):
        self.ors_handler = ors.ORSHandler(api_key)

    def _identify_gush_dan_customers(self, coords: List[Tuple[float, float]]) -> Set[int]:
        """
        Identify customer nodes that fall within Gush Dan bounds.
        Node 0 is the depot, so we skip it.
        
        Returns:
            Set of node indices (excluding depot) that are in Gush Dan
        """
        gush_dan_indices = set()
        bounds = Config.GUSH_DAN_BOUNDS
        
        for node_idx, (lat, lng) in enumerate(coords):
            # Skip depot (node 0)
            if node_idx == 0:
                continue
            
            # Check if coordinates fall within Gush Dan bounds
            if (bounds['min_lat'] <= lat <= bounds['max_lat'] and
                bounds['min_lon'] <= lng <= bounds['max_lon']):
                gush_dan_indices.add(node_idx)
        
        return gush_dan_indices

    def solve(self,
              locations: List[str], 
              demands: List[int], 
              params: OptimizationParams, 
              coords: List[Tuple[float, float]]) -> Solution:
        """
        Executes the optimization pipeline using OR-Tools global solver.
        
        Args:
            locations: List of location identifiers
            demands: List of scaled volume demands (m³ * 1000 as integers). 
                     Index 0 should be 0 (depot), customer nodes have scaled effective volumes.
            params: Optimization parameters including scaled capacities (small_capacity, big_capacity)
            coords: List of (lat, lng) coordinates for each location
        """
        logger.info(f"Starting Optimization for {len(locations)} locations.")
        
        try:
            # --- Phase 0: Data Prep & Matrix ---
            if not coords:
                raise ValueError("No coordinates provided.")
            
            logger.info("Phase 0: Fetching Distance Matrix (with caching)...")
            matrix_data = self.ors_handler.get_distance_matrix(coords)
            
            # --- Phase 0.5: Identify Gush Dan Customers ---
            gush_dan_node_indices = self._identify_gush_dan_customers(coords)
            logger.info(f"Identified {len(gush_dan_node_indices)} Gush Dan customers (must use Small Trucks)")
            
            # --- Phase 0.6: Setup Heterogeneous Fleet ---
            num_small_trucks = params.get('num_small_trucks', 6)
            num_big_trucks = params.get('num_big_trucks', 12)
            small_capacity = params.get('small_capacity', params.get('capacity', 0))
            big_capacity = params.get('big_capacity', params.get('capacity', 0))
            
            # Construct vehicle capacities list: first num_small_trucks are Small, rest are Big
            vehicle_capacities = [small_capacity] * num_small_trucks + [big_capacity] * num_big_trucks
            total_fleet_size = len(vehicle_capacities)
            
            # Track which vehicle indices are Small Trucks (0 to num_small_trucks-1)
            small_truck_vehicle_indices = list(range(num_small_trucks))
            
            logger.info(f"Fleet composition: {num_small_trucks} Small Trucks (scaled volume capacity={small_capacity}), {num_big_trucks} Big Trucks (scaled volume capacity={big_capacity})")
            
            # Build data model for OR-Tools
            data = {
                'distance_matrix': matrix_data['distances'],
                'time_matrix': matrix_data['durations'],
                'demands': demands,  # Scaled volume demands (m³ * 1000). Node 0 is warehouse with demand 0
                'vehicle_capacities': vehicle_capacities,  # Scaled volume capacities (m³ * 1000)
                'num_vehicles': total_fleet_size,
                'depot': 0,
                'max_shift_seconds': params['max_shift_seconds'],
                'gush_dan_node_indices': gush_dan_node_indices,
                'small_truck_vehicle_indices': small_truck_vehicle_indices
            }
            
            # Solve with OR-Tools
            logger.info("Phase 1: Solving with OR-Tools...")
            solution = self._solve_with_ortools(data, coords, locations, params)
            
            # Store matrix data in solution for reuse (avoid duplicate API calls)
            solution['matrix_data'] = {
                'distances': matrix_data['distances'],
                'durations': matrix_data['durations']
            }
            
            return solution

        except Exception as e:
            logger.error(f"Optimization Failed: {e}")
            raise

    def _solve_with_ortools(self, data: Dict[str, Any], coords: List[Tuple[float, float]], 
                            locations: List[str], params: OptimizationParams) -> Solution:
        """
        Solve the VRP using OR-Tools RoutingModel.
        
        Args:
            data: Dictionary containing distance_matrix, time_matrix, demands, vehicle_capacities, etc.
            coords: List of (lat, lng) coordinates
            locations: List of location identifiers
            params: Optimization parameters
            
        Returns:
            Solution dictionary with routes, metrics, and unserved customers
        """
        # Validate required parameters
        if 'service_time_seconds' not in params:
            logger.warning("service_time_seconds not found in params, using default 300 seconds (5 minutes)")
            params['service_time_seconds'] = 300
        
        num_nodes = len(data['distance_matrix'])
        num_vehicles = data['num_vehicles']
        depot = data['depot']
        
        # Log solver configuration
        service_time = params.get('service_time_seconds', 300)
        logger.info(f"Solver configuration: {num_vehicles} vehicles, max_shift={data['max_shift_seconds']}s, service_time={service_time}s")
        
        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot)
        
        # Create routing model
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback
        def distance_callback(from_index: int, to_index: int) -> int:
            """Returns the distance between the two nodes."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(data['distance_matrix'][from_node][to_node])
        
        distance_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        # Cost callback - weighted combination of distance and time
        def cost_callback(from_index: int, to_index: int) -> int:
            """
            Returns weighted cost: (Distance * 1.0) + (Time * 1.0).
            Uses travel time only (not service time) since service time is fixed per customer.
            """
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            distance = int(data['distance_matrix'][from_node][to_node])
            travel_time = int(data['time_matrix'][from_node][to_node])
            return int(distance * 1.0 + travel_time * 6.0)
        
        cost_callback_index = routing.RegisterTransitCallback(cost_callback)
        
        # Set arc cost evaluator
        routing.SetArcCostEvaluatorOfAllVehicles(cost_callback_index)

        # Set fixed cost per vehicle to enforce fleet size minimization (Stage 2 priority)
        routing.SetFixedCostOfAllVehicles(Config.VEHICLE_FIXED_COST)
        
        # Time callback - includes service time when leaving customer nodes
        # Service time is added when leaving a customer, but NOT when leaving the depot
        service_time_seconds = params.get('service_time_seconds', 300)
        
        def time_callback(from_index: int, to_index: int) -> int:
            """
            Returns the total time cost: travel time + service time (if leaving a customer).
            
            Rule:
            - If leaving the depot: return only travel time
            - If leaving a customer: return travel time + service time
            """
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            # Get base travel time from matrix
            travel_time = int(data['time_matrix'][from_node][to_node])
            
            # Add service time if leaving a customer node (not the depot)
            if from_node != depot:
                total_time = travel_time + service_time_seconds
            else:
                total_time = travel_time
            
            return int(total_time)
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        
        # Add time dimension with soft constraints (elastic walls) - 48 hour horizon
        # Allows solver to exceed time limits with penalties
        # 1. Add the dimension (returns bool)
        routing.AddDimension(
            time_callback_index,
            0,  # slack
            172800,  # horizon (48 hours for soft constraints)
            True,  # fix_start_cumul_to_zero
            "Time"
        )

        # 2. Retrieve the dimension object explicitly
        time_dimension = routing.GetDimensionOrDie("Time")

        # Add soft upper bounds for time (elastic constraints)
        for vehicle_id in range(data['num_vehicles']):
            index = routing.End(vehicle_id)
            time_dimension.SetCumulVarSoftUpperBound(index, data['max_shift_seconds'], 100_000)

        # Get the Time dimension and set global span cost to balance workloads across drivers
        time_dim = routing.GetDimensionOrDie("Time")
        time_dim.SetGlobalSpanCostCoefficient(100)
        
        # Demand callback - returns the scaled volume demand for each node
        def demand_callback(from_index: int) -> int:
            """
            Returns the scaled volume demand of the node (m³ * 1000 as integer).
            
            Note: Node 0 (depot) should have demand 0.
            Customer nodes have their scaled effective volume (force_volume / safety_factor * 1000).
            """
            from_node = manager.IndexToNode(from_index)
            demand = data['demands'][from_node]
            
            # Ensure demand is non-negative and convert to int
            demand_value = max(0, int(demand))
            
            # Validate depot has zero demand
            if from_node == depot and demand_value != 0:
                logger.warning(f"Depot (node {depot}) has non-zero demand: {demand_value}. Setting to 0.")
                return 0
            
            return demand_value
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        # Add capacity dimension with soft constraints (elastic walls)
        # 1. Add the dimension (returns bool, discard result)
        routing.AddDimension(
            demand_callback_index,
            0,  # slack
            1000000,  # capacity (infinite for soft constraints)
            True,  # fix_start_cumul_to_zero
            "Capacity"
        )

        # 2. Retrieve the dimension object explicitly
        capacity_dimension = routing.GetDimensionOrDie("Capacity")

        # Add soft upper bounds for capacity (elastic constraints)
        for vehicle_id in range(data['num_vehicles']):
            index = routing.End(vehicle_id)
            original_vehicle_capacity = data['vehicle_capacities'][vehicle_id]
            capacity_dimension.SetCumulVarSoftUpperBound(index, original_vehicle_capacity, 100_000)
        
        # Apply Gush Dan constraints: restrict Gush Dan customers to Small Trucks only
        gush_dan_node_indices = data.get('gush_dan_node_indices', set())
        small_truck_vehicle_indices = data.get('small_truck_vehicle_indices', [])
        
        if gush_dan_node_indices and small_truck_vehicle_indices:
            logger.info(f"Applying Gush Dan constraints: {len(gush_dan_node_indices)} customers restricted to Small Trucks")
            for gush_dan_node in gush_dan_node_indices:
                node_index = manager.NodeToIndex(gush_dan_node)
                # Restrict this node to only be assigned to Small Truck vehicles
                routing.VehicleVar(node_index).SetValues(small_truck_vehicle_indices)
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 300  # 300 second time limit for greedy packing
        
        # Solve
        logger.info("Solving VRP with OR-Tools...")
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution is None:
            logger.error("OR-Tools failed to find a solution")
            return {
                'solution_found': False,
                'routes': [],
                'route_metrics': [],
                'unserved': [
                    {'id': i, 'reason': 'Solver failed to find solution', 'demand': data['demands'][i]}
                    for i in range(1, num_nodes)
                ],
                'suggestions': {
                    'all_served': False,
                    'message': 'Solver failed to find a feasible solution',
                    'suggestions': []
                },
                'metrics': {
                    'total_distance': 0,
                    'total_time': 0,
                    'max_route_time': 0,
                    'num_vehicles_used': 0,
                    'customers_served': 0,
                    'customers_unserved': num_nodes - 1,
                    'service_rate': 0
                }
            }
        
        # Extract solution
        return self._extract_solution(routing, manager, solution, data, coords, locations, params)

    def _extract_solution(self, routing: pywrapcp.RoutingModel, manager: pywrapcp.RoutingIndexManager,
                         solution: pywrapcp.Assignment, data: Dict[str, Any], 
                         coords: List[Tuple[float, float]], locations: List[str],
                         params: OptimizationParams) -> Solution:
        """
        Extract routes from OR-Tools solution and hydrate with ORS directions.
        
        Args:
            routing: OR-Tools routing model
            manager: OR-Tools index manager
            solution: OR-Tools solution assignment
            data: Data dictionary with matrices and parameters
            coords: List of (lat, lng) coordinates
            locations: List of location identifiers
            params: Optimization parameters
            
        Returns:
            Solution dictionary with routes, metrics, and unserved customers
        """
        routes = []
        route_metrics = []
        served_indices = set()
        
        # Determine truck type based on vehicle index
        num_small_trucks = len(data.get('small_truck_vehicle_indices', []))
        
        # Extract routes for each vehicle
        for vehicle_id in range(data['num_vehicles']):
            route_nodes = []
            index = routing.Start(vehicle_id)
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index != data['depot']:  # Don't add depot in the middle
                    route_nodes.append(node_index)
                    served_indices.add(node_index)
                index = solution.Value(routing.NextVar(index))
            
            # Only add non-empty routes
            if route_nodes:
                # Build full route: depot -> customers -> depot
                full_route = [data['depot']] + route_nodes + [data['depot']]
                
                # Determine vehicle type: first num_small_trucks are Small, rest are Big
                vehicle_type = "Small" if vehicle_id < num_small_trucks else "Big"
                
                # Get route coordinates for directions API
                route_coords = [coords[node] for node in full_route]
                
                # Get geometry and precise metrics from ORS Directions API
                try:
                    directions_data = self.ors_handler.get_directions(route_coords)
                    geometry = directions_data['geometry']
                    distance = directions_data['distance']
                    duration = directions_data['duration']
                    
                    # Add service time for each customer stop
                    service_time = params.get('service_time_seconds', 300)
                    duration += len(route_nodes) * service_time
                    
                    # Format polylines for frontend (list of coordinate lists)
                    polylines = [geometry] if geometry else []
                    
                    # Store route as list (for backward compatibility) and add vehicle_type to metrics
                    routes.append(full_route)
                    route_metrics.append({
                        'distance': distance,
                        'duration': duration,
                        'polylines': polylines,
                        'vehicle_type': vehicle_type,
                        'vehicle_id': vehicle_id
                    })
                except Exception as e:
                    logger.warning(f"Failed to get directions for vehicle {vehicle_id}: {e}. Using matrix estimates.")
                    # Fallback: calculate from matrix
                    total_distance = 0
                    total_duration = 0
                    for i in range(len(full_route) - 1):
                        from_node = full_route[i]
                        to_node = full_route[i + 1]
                        total_distance += data['distance_matrix'][from_node][to_node]
                        total_duration += data['time_matrix'][from_node][to_node]
                    
                    service_time = params.get('service_time_seconds', 300)
                    total_duration += len(route_nodes) * service_time
                    
                    route_metrics.append({
                        'distance': total_distance,
                        'duration': total_duration,
                        'polylines': [],  # No geometry available
                        'vehicle_type': vehicle_type,
                        'vehicle_id': vehicle_id
                    })
        
        # Identify unserved customers
        all_customer_indices = set(range(1, len(locations)))
        unserved_indices = all_customer_indices - served_indices
        
        # Get maximum capacity (big truck capacity) for validation
        max_capacity = max(data['vehicle_capacities']) if data['vehicle_capacities'] else params.get('big_capacity', params.get('capacity', 0))
        
        unserved_customers = []
        for idx in unserved_indices:
            # Check if scaled volume demand exceeds maximum scaled capacity
            if data['demands'][idx] > max_capacity:
                reason = 'Volume demand exceeds vehicle capacity'
            else:
                reason = 'Not assigned to any route'
            
            unserved_customers.append({
                'id': idx,
                'reason': reason,
                'demand': data['demands'][idx]
            })
        
        # Optional heuristic retry for remaining unserved customers
        if unserved_customers and len(routes) < data['num_vehicles']:
            logger.info(f"Attempting heuristic fallback for {len(unserved_customers)} unserved customers")
            remaining_unserved = [u['id'] for u in unserved_customers if u['demand'] <= max_capacity]
            
            if remaining_unserved:
                # Try to create additional routes using geometric fallback
                new_routes, new_metrics, newly_served = self._apply_heuristic_fallback(
                    remaining_unserved, coords, data['demands'], params, data
                )
            
            if new_routes:
                    routes.extend(new_routes)
                    route_metrics.extend(new_metrics)
                    served_indices.update(newly_served)
                    # Update unserved list
                    unserved_customers = [u for u in unserved_customers if u['id'] not in newly_served]
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            unserved_customers, 
            data['demands'],
            params, 
            coords, 
            len(served_indices),
            len(all_customer_indices)
        )

        # Calculate metrics
        total_distance = sum(r['distance'] for r in route_metrics)
        total_time = sum(r['duration'] for r in route_metrics)
        max_route_time = max((r['duration'] for r in route_metrics), default=0)

        return {
            'solution_found': True,
            'routes': routes,
            'route_metrics': route_metrics,
            'unserved': unserved_customers,
            'suggestions': suggestions,
            'metrics': {
                'total_distance': total_distance,
                'total_time': total_time,
                'max_route_time': max_route_time,
                'num_vehicles_used': len(routes),
                'customers_served': len(served_indices),
                'customers_unserved': len(unserved_customers),
                'service_rate': len(served_indices) / len(all_customer_indices) if all_customer_indices else 0
            }
        }

    def _apply_heuristic_fallback(self, unserved_indices: List[int], coords: List[Tuple[float, float]],
                                  demands: List[int], params: OptimizationParams, data: Dict[str, Any]) -> Tuple[List[List[int]], List[Dict], Set[int]]:
        """
        Apply geometric fallback for unserved customers.
        Returns (routes, metrics, served_indices).
        """
        if not unserved_indices:
            return [], [], set()
        
        def get_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        routes = []
        metrics_list = []
        served = set()
        
        # Sort by distance from warehouse
        warehouse_coord = coords[0]
        unserved_sorted = sorted(unserved_indices, key=lambda idx: get_dist(coords[idx], warehouse_coord))
        
        current_route = []
        current_load = 0  # Scaled volume load (m³ * 1000)
        
        for idx in unserved_sorted:
            # Compare scaled volume demand against scaled capacity
            if current_load + demands[idx] <= params.get('capacity', params.get('big_capacity', 0)):
                current_route.append(idx)
                current_load += demands[idx]
            else:
                if current_route:
                    # Finalize current route
                    full_route = [0] + current_route + [0]
                    routes.append(full_route)
                    served.update(current_route)
                    
                    # Get directions for geometry
                    route_coords = [coords[node] for node in full_route]
                    try:
                        directions_data = self.ors_handler.get_directions(route_coords)
                        service_time = params.get('service_time_seconds', 300)
                        metrics_list.append({
                            'distance': directions_data['distance'],
                            'duration': directions_data['duration'] + len(current_route) * service_time,
                            'polylines': [directions_data['geometry']] if directions_data['geometry'] else []
                        })
                    except Exception as e:
                        logger.warning(f"Failed to get directions for fallback route: {e}")
                        # Use matrix estimates
                        total_dist = sum(data['distance_matrix'][full_route[i]][full_route[i+1]] 
                                       for i in range(len(full_route) - 1))
                        total_time = sum(data['time_matrix'][full_route[i]][full_route[i+1]] 
                                       for i in range(len(full_route) - 1))
                        service_time = params.get('service_time_seconds', 300)
                        metrics_list.append({
                            'distance': total_dist,
                            'duration': total_time + len(current_route) * service_time,
                            'polylines': []
                        })
                
                # Start new route
                current_route = [idx]
                current_load = demands[idx]
        
        # Add final route if exists
        if current_route:
            full_route = [0] + current_route + [0]
            routes.append(full_route)
            served.update(current_route)
            
            route_coords = [coords[node] for node in full_route]
            try:
                directions_data = self.ors_handler.get_directions(route_coords)
                service_time = params.get('service_time_seconds', 300)
                metrics_list.append({
                    'distance': directions_data['distance'],
                    'duration': directions_data['duration'] + len(current_route) * service_time,
                    'polylines': [directions_data['geometry']] if directions_data['geometry'] else []
                })
            except Exception as e:
                logger.warning(f"Failed to get directions for final fallback route: {e}")
                total_dist = sum(data['distance_matrix'][full_route[i]][full_route[i+1]] 
                               for i in range(len(full_route) - 1))
                total_time = sum(data['time_matrix'][full_route[i]][full_route[i+1]] 
                               for i in range(len(full_route) - 1))
                service_time = params.get('service_time_seconds', 300)
                metrics_list.append({
                    'distance': total_dist,
                    'duration': total_time + len(current_route) * service_time,
                    'polylines': []
                })
        
        return routes, metrics_list, served

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
        
        # Check capacity constraints - use maximum scaled capacity (big truck)
        max_capacity = params.get('big_capacity', params.get('capacity', 0))
        if any(d > max_capacity for d in unserved_demands):
            oversize_customers = [
                u for u in unserved_customers 
                if u['demand'] > max_capacity
            ]
            suggestions.append({
                'type': 'capacity_issue',
                'message': f"{len(oversize_customers)} customers exceed vehicle volume capacity",
                'details': [{
                    'customer_id': u['id'],
                    'scaled_volume_demand': u['demand'],
                    'scaled_vehicle_capacity': max_capacity
                } for u in oversize_customers],
                'suggestion': 'Consider using vehicles with higher volume capacity or splitting orders'
            })

        # Check time constraints
        max_possible_routes = params.get('fleet_size', 18)
        if len(unserved_customers) > 0 and len(unserved_customers) <= max_possible_routes:
            # Estimate current routes used (we don't have exact count here, use fleet_size as proxy)
            current_routes_estimate = max_possible_routes  # Conservative estimate
            suggestions.append({
                'type': 'fleet_adjustment',
                'message': f'Adding {len(unserved_customers)} more routes could serve all customers',
                'suggestion': f"Consider increasing fleet size or adjusting constraints to accommodate {len(unserved_customers)} additional customers"
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

