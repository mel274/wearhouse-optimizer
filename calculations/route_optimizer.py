"""
Main orchestration for the Warehouse Route Optimization.
Version: 4.0 (PyVroom Integration)
"""
import logging
import time
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Set

# PyVroom imports (replaces OR-Tools)
import vroom
from vroom import Input as VroomInput, Vehicle, Job, Amount, TimeWindow

from .types import Solution, OptimizationParams, UnservedCustomer
from . import ors, metrics
import shared.utils as utils
from shared.geo_utils import identify_gush_dan_customers
from config import Config

logger = logging.getLogger(__name__)

class RouteOptimizer:
    """
    Orchestrates the optimization pipeline using PyVroom:
    1. Fetch Matrix (cached) -> 2. PyVroom Solver -> 3. Hydrate with ORS Directions
    """
    
    def __init__(self, api_key: str):
        self.ors_handler = ors.ORSHandler(api_key)

    def _identify_gush_dan_customers(self, coords: List[Tuple[float, float]]) -> Set[int]:
        """
        Identify customer nodes that fall within Gush Dan bounds.
        
        Delegates to shared/geo_utils.py to avoid drift between RouteOptimizer and ClusterService.
        
        Returns:
            Set of node indices (excluding depot) that are in Gush Dan
        """
        return identify_gush_dan_customers(coords)

    # ========== PyVroom Model Builders (Phase 2) ==========

    def _build_vroom_cost_matrix(self, data: Dict[str, Any]) -> List[List[int]]:
        """
        Build the cost matrix for VROOM optimization.
        
        Cost formula: cost[i][j] = int(distance[i][j] + 6 * duration[i][j])
        VROOM expects integer costs.
        
        Args:
            data: Dictionary containing 'distance_matrix' and 'time_matrix'
            
        Returns:
            2D list of integer costs
        """
        distance_matrix = data['distance_matrix']
        time_matrix = data['time_matrix']
        num_nodes = len(distance_matrix)
        
        cost_matrix = []
        for i in range(num_nodes):
            row = []
            for j in range(num_nodes):
                distance = distance_matrix[i][j]
                duration = time_matrix[i][j]
                # Cost formula per plan.md: distance + 6 * duration
                cost = int(distance + 6 * duration)
                row.append(cost)
            cost_matrix.append(row)
        
        return cost_matrix

    def _build_vroom_jobs(self, data: Dict[str, Any], params: OptimizationParams) -> List[Job]:
        """
        Build VROOM Job objects for all customers (nodes 1..N, excluding depot).
        
        Args:
            data: Dictionary containing 'demands', 'gush_dan_node_indices'
            params: Contains 'service_time_seconds'
            
        Returns:
            List of vroom.Job objects
        """
        demands = data['demands']
        gush_dan_node_indices = data.get('gush_dan_node_indices', set())
        service_time_seconds = params.get('service_time_seconds', 300)
        num_nodes = len(demands)
        
        jobs = []
        # Nodes 1..N are customers (node 0 is depot)
        for node_idx in range(1, num_nodes):
            # Skill 1 = Gush Dan restriction (only small trucks can serve)
            skills = {1} if node_idx in gush_dan_node_indices else set()
            
            job = Job(
                id=node_idx,
                location=node_idx,  # Location index in the matrix
                delivery=Amount([demands[node_idx]]),  # Single dimension: volume
                service=service_time_seconds,
                skills=skills
            )
            jobs.append(job)
        
        return jobs

    def _build_vroom_vehicles(self, data: Dict[str, Any], vehicle_limit: int) -> List[Vehicle]:
        """
        Build VROOM Vehicle objects up to the specified limit.
        
        Args:
            data: Dictionary containing 'vehicle_capacities', 'small_truck_vehicle_indices', 
                  'max_shift_seconds', 'depot'
            vehicle_limit: Maximum number of vehicles to create
            
        Returns:
            List of vroom.Vehicle objects
        """
        vehicle_capacities = data['vehicle_capacities'][:vehicle_limit]
        small_truck_indices = set(data.get('small_truck_vehicle_indices', []))
        max_shift_seconds = data['max_shift_seconds']
        depot = data['depot']
        
        vehicles = []
        for vehicle_id in range(vehicle_limit):
            capacity = vehicle_capacities[vehicle_id]
            
            # Skill 1 = can serve Gush Dan customers (only small trucks have this)
            skills = {1} if vehicle_id in small_truck_indices else set()
            
            vehicle = Vehicle(
                id=vehicle_id,
                start=depot,  # Start at depot
                end=depot,    # Return to depot
                capacity=Amount([capacity]),  # Single dimension: volume
                max_travel_time=max_shift_seconds,  # Enforce shift limit
                skills=skills
            )
            vehicles.append(vehicle)
        
        return vehicles

    def _build_vroom_input(self, data: Dict[str, Any], params: OptimizationParams, 
                           vehicle_limit: int) -> Tuple[VroomInput, List[List[int]]]:
        """
        Build the complete VROOM Input object with cost matrix, jobs, and vehicles.
        
        Args:
            data: Dictionary containing matrices, demands, capacities, etc.
            params: Optimization parameters
            vehicle_limit: Maximum number of vehicles to use
            
        Returns:
            Tuple of (VroomInput object, cost_matrix for reference)
        """
        # Build cost matrix
        cost_matrix = self._build_vroom_cost_matrix(data)
        
        # Build jobs (customers)
        jobs = self._build_vroom_jobs(data, params)
        
        # Build vehicles
        vehicles = self._build_vroom_vehicles(data, vehicle_limit)
        
        # Create VROOM Input
        vroom_input = VroomInput()
        
        # Must use np.uintc (not np.uint32) to get 'I' buffer format required by pyvroom on Windows
        # Create _vroom.Matrix directly to bypass the wrapper's dtype conversion
        from vroom import _vroom
        
        # Set the actual time matrix as durations (for time-based constraints like max_travel_time)
        time_matrix = data['time_matrix']
        time_matrix_np = np.array(time_matrix, dtype=np.uintc)
        durations_vroom_matrix = _vroom.Matrix(time_matrix_np)
        vroom_input.set_durations_matrix(profile="car", matrix_input=durations_vroom_matrix)
        
        # Set the custom cost matrix for optimization objective (distance + 6 * duration)
        cost_matrix_np = np.array(cost_matrix, dtype=np.uintc)
        costs_vroom_matrix = _vroom.Matrix(cost_matrix_np)
        vroom_input.set_costs_matrix(profile="car", matrix_input=costs_vroom_matrix)
        
        # Add all jobs
        for job in jobs:
            vroom_input.add_job(job)
        
        # Add all vehicles
        for vehicle in vehicles:
            vroom_input.add_vehicle(vehicle)
        
        logger.info(f"Built VROOM input: {len(jobs)} jobs, {len(vehicles)} vehicles, "
                   f"{len(cost_matrix)}x{len(cost_matrix)} cost matrix")
        
        return vroom_input, cost_matrix

    # ========== End PyVroom Model Builders ==========

    def solve(self,
              locations: List[str], 
              demands: List[int], 
              params: OptimizationParams, 
              coords: List[Tuple[float, float]],
              min_trucks_override: int = None,
              initial_routes: List[List[int]] = None) -> Solution:
        """
        Executes the optimization pipeline using PyVroom solver.
        
        Args:
            locations: List of location identifiers
            demands: List of scaled volume demands (m³ * 1000 as integers). 
                     Index 0 should be 0 (depot), customer nodes have scaled effective volumes.
            params: Optimization parameters including scaled capacities (small_capacity, big_capacity)
            coords: List of (lat, lng) coordinates for each location
            min_trucks_override: If provided, Fleet Squeeze starts from this minimum instead of calculated minimum
            initial_routes: Optional seeded routes from ClusterService (list of customer indices per vehicle, NO DEPOT)
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
            
            # Calculate theoretical minimum trucks (Fleet Squeeze)
            total_volume = sum(demands[1:]) / 1000.0  # Convert back from scaled units
            avg_truck_capacity = ((small_capacity / 1000.0) + (big_capacity / 1000.0)) / 2.0  # Convert back to raw m³

            # Detect unit mismatch: if volume >> capacity and capacity is small (m³ vs liters)
            if total_volume > avg_truck_capacity and avg_truck_capacity < 1000:
                # Volume is likely in liters, capacity in m³ - convert capacity to liters for comparison
                avg_truck_capacity *= 1000
                logger.info(f"Detected unit mismatch - converting capacity from m³ to liters for calculation")

            min_trucks = max(1, math.ceil(total_volume / avg_truck_capacity))
            # Safety clamp: never exceed available fleet size
            min_trucks = min(min_trucks, len(vehicle_capacities))

            logger.info(f"Fleet Squeeze: total_volume={total_volume:.1f}m³, avg_capacity={avg_truck_capacity:.1f}m³, theoretical_min={min_trucks} trucks")

            # Apply min_trucks_override if provided (used by exception budget retry)
            start_trucks = min_trucks
            if min_trucks_override is not None and min_trucks_override > min_trucks:
                start_trucks = min(min_trucks_override, len(vehicle_capacities))
                logger.info(f"Min trucks override: starting from {start_trucks} trucks (override={min_trucks_override})")

            # Iterative Squeeze Loop: Start with theoretical minimum, increment until solution found
            solution = None
            max_fleet_size = len(vehicle_capacities)

            for n_trucks in range(start_trucks, max_fleet_size + 1):
                logger.info(f"Attempting solve with {n_trucks} trucks...")
                solution = self._solve_with_ortools(
                    data, coords, locations, params,
                    vehicle_limit=n_trucks,
                    initial_routes=initial_routes  # Pass seed to each attempt
                )

                if solution and solution.get('solution_found'):
                    logger.info(f"Success with {n_trucks} trucks!")
                    break  # Exit loop immediately on first success

            # If no solution found with any fleet size, return failure
            if not solution or not solution.get('solution_found'):
                logger.error("Fleet Squeeze failed: No solution found with any fleet size")
                return {
                    'solution_found': False,
                    'routes': [],
                    'route_metrics': [],
                    'unserved': [{'id': i, 'reason': 'Solver failed to find solution', 'demand': demands[i]} for i in range(1, len(demands))],
                    'suggestions': {'all_served': False, 'message': 'Solver failed to find a feasible solution', 'suggestions': []},
                    'metrics': {'total_distance': 0, 'total_time': 0, 'max_route_time': 0, 'num_vehicles_used': 0, 'customers_served': 0, 'customers_unserved': len(demands) - 1, 'service_rate': 0}
                }
            
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
                            locations: List[str], params: OptimizationParams, vehicle_limit: int,
                            initial_routes: List[List[int]] = None) -> Solution:
        """
        Solve the VRP using PyVroom.
        
        Args:
            data: Dictionary containing distance_matrix, time_matrix, demands, vehicle_capacities, etc.
            coords: List of (lat, lng) coordinates
            locations: List of location identifiers
            params: Optimization parameters
            vehicle_limit: Maximum number of vehicles to use
            initial_routes: Optional seeded routes (list of customer indices per vehicle, NO DEPOT)
                           Note: PyVroom does not support initial routes - this parameter is ignored.
            
        Returns:
            Solution dictionary with routes, metrics, and unserved customers
        """
        # Validate required parameters
        if 'service_time_seconds' not in params:
            logger.warning("service_time_seconds not found in params, using default 300 seconds (5 minutes)")
            params['service_time_seconds'] = 300
        
        num_nodes = len(data['distance_matrix'])
        depot = data['depot']
        
        # Log solver configuration
        service_time = params.get('service_time_seconds', 300)
        logger.info(f"Solver configuration: {vehicle_limit} vehicles (limited), max_shift={data['max_shift_seconds']}s, service_time={service_time}s")
        data['vehicle_limit'] = vehicle_limit
        
        # Build VROOM input using Phase 2 helpers
        try:
            vroom_input, cost_matrix = self._build_vroom_input(data, params, vehicle_limit)
        except Exception as e:
            logger.error(f"Failed to build VROOM input: {e}")
            return {
                'solution_found': False,
                'routes': [],
                'route_metrics': [],
                'unserved': [
                    {'id': i, 'reason': f'Failed to build solver input: {e}', 'demand': data['demands'][i]}
                    for i in range(1, num_nodes)
                ],
                'suggestions': {
                    'all_served': False,
                    'message': 'Failed to build solver input',
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
        
        # Solve with PyVroom
        logger.info("Solving VRP with PyVroom...")
        try:
            vroom_solution = vroom_input.solve(exploration_level=2, nb_threads=4)
        except Exception as e:
            logger.error(f"PyVroom solve failed: {e}")
            return {
                'solution_found': False,
                'routes': [],
                'route_metrics': [],
                'unserved': [
                    {'id': i, 'reason': f'Solver failed: {e}', 'demand': data['demands'][i]}
                    for i in range(1, num_nodes)
                ],
                'suggestions': {
                    'all_served': False,
                    'message': f'PyVroom solver failed: {e}',
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
        
        # Extract solution from PyVroom result
        return self._extract_vroom_solution(vroom_solution, data, coords, locations, params)

    def _extract_vroom_solution(self, vroom_solution: Any, data: Dict[str, Any],
                                coords: List[Tuple[float, float]], locations: List[str],
                                params: OptimizationParams) -> Solution:
        """
        Extract routes from PyVroom solution and hydrate with ORS directions.
        
        Produces the exact same Solution schema as the original OR-Tools implementation.
        
        Args:
            vroom_solution: PyVroom solution object (routes as pandas DataFrame)
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
        
        # Track coordinates that couldn't be routed (for UI warning)
        all_skipped_coords = []
        
        # Track node indices that were skipped during diagnostic (for UI table)
        all_skipped_node_indices = []
        
        # Determine truck type based on vehicle index
        num_small_trucks = len(data.get('small_truck_vehicle_indices', []))
        depot = data['depot']
        
        # Extract routes from PyVroom solution (DataFrame format)
        routes_df = vroom_solution.routes
        
        # Get unique vehicle IDs that have routes
        if len(routes_df) > 0:
            vehicle_ids = routes_df['vehicle_id'].unique()
        else:
            vehicle_ids = []
        
        for vehicle_id in vehicle_ids:
            # Filter steps for this vehicle
            vehicle_steps = routes_df[routes_df['vehicle_id'] == vehicle_id]
            
            # Extract job IDs (type == 'job')
            job_steps = vehicle_steps[vehicle_steps['type'] == 'job']
            route_nodes = job_steps['id'].tolist()
            
            # Add served indices
            for job_id in route_nodes:
                served_indices.add(job_id)
            
            # Only add non-empty routes
            if route_nodes:
                # Build full route: depot -> customers -> depot
                full_route = [depot] + route_nodes + [depot]
                
                # Determine vehicle type: first num_small_trucks are Small, rest are Big
                vehicle_type = "Small" if vehicle_id < num_small_trucks else "Big"
                
                # Get route coordinates for directions API
                route_coords = [coords[node] for node in full_route]
                
                # Get geometry and precise metrics from ORS Directions API
                directions_data = self.ors_handler.get_directions(route_coords)
                
                if directions_data is not None:
                    # Successful ORS response - use detailed geometry and metrics
                    geometry = directions_data['geometry']
                    distance = directions_data['distance']
                    duration = directions_data['duration']
                    
                    # Collect any skipped coordinates (unroutable points)
                    skipped_coords = directions_data.get('skipped_coordinates', [])
                    if skipped_coords:
                        all_skipped_coords.extend(skipped_coords)
                    
                    # Add service time for each customer stop
                    service_time = params.get('service_time_seconds', 300)
                    duration += len(route_nodes) * service_time
                    
                    # Format polylines for frontend (list of coordinate lists)
                    polylines = [geometry] if geometry else []
                
                else:
                    # Fallback to matrix-based estimates
                    logger.warning(f"ORS Directions API failed for Vehicle {vehicle_id}, using matrix fallback")
                    
                    # Create straight-line geometry connecting stops in order
                    geometry = [(coords[node][0], coords[node][1]) for node in full_route]
                    
                    # Calculate metrics from internal matrices
                    total_distance = 0.0
                    total_duration = 0.0
                    for i in range(len(full_route) - 1):
                        from_node = full_route[i]
                        to_node = full_route[i + 1]
                        total_distance += data['distance_matrix'][from_node][to_node]
                        total_duration += data['time_matrix'][from_node][to_node]
                    
                    # Add service time for each customer stop
                    service_time = params.get('service_time_seconds', 300)
                    total_duration += len(route_nodes) * service_time
                    
                    distance = total_distance
                    duration = total_duration
                    polylines = [geometry]
                
                # Store route and metrics
                routes.append(full_route)
                route_metrics.append({
                    'distance': distance,
                    'duration': duration,
                    'polylines': polylines,
                    'vehicle_type': vehicle_type,
                    'vehicle_id': int(vehicle_id)
                })
        
        # Identify unserved customers from PyVroom unassigned jobs
        all_customer_indices = set(range(1, len(locations)))
        unserved_indices = all_customer_indices - served_indices
        
        # Get maximum capacity (big truck capacity) for validation
        max_capacity = max(data['vehicle_capacities']) if data['vehicle_capacities'] else params.get('big_capacity', params.get('capacity', 0))
        
        unserved_customers = []
        
        # Map unassigned jobs from PyVroom solution (list of Job objects)
        unassigned_jobs = vroom_solution.unassigned
        if len(unassigned_jobs) > 0:
            for job in unassigned_jobs:
                job_id = job._id  # Job ID is stored in _id attribute
                # Determine reason based on constraints
                if data['demands'][job_id] > max_capacity:
                    reason = 'Volume demand exceeds vehicle capacity'
                else:
                    reason = 'Not assigned to any route (constraint violation)'
                
                unserved_customers.append({
                    'id': job_id,
                    'reason': reason,
                    'demand': data['demands'][job_id]
                })
        
        # Also add any jobs that weren't in the solution at all
        for idx in unserved_indices:
            if not any(u['id'] == idx for u in unserved_customers):
                if data['demands'][idx] > max_capacity:
                    reason = 'Volume demand exceeds vehicle capacity'
                else:
                    reason = 'Not assigned to any route'
                unserved_customers.append({
                    'id': idx,
                    'reason': reason,
                    'demand': data['demands'][idx]
                })
        
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
        
        # Map skipped coordinates to node indices for UI display
        skipped_node_indices = list(all_skipped_node_indices)
        
        for skipped_coord in all_skipped_coords:
            for node_idx, node_coord in enumerate(coords):
                if (abs(node_coord[0] - skipped_coord[0]) < 0.0001 and
                    abs(node_coord[1] - skipped_coord[1]) < 0.0001):
                    if node_idx != 0 and node_idx not in skipped_node_indices:
                        skipped_node_indices.append(node_idx)
                    break
        
        if skipped_node_indices:
            logger.warning(f"Skipped {len(skipped_node_indices)} unroutable customers during directions fetch")
        
        # Fleet Squeeze: treat any unassigned jobs as infeasible for this truck count
        # A solution is only "found" if ALL jobs are assigned (no unserved customers)
        solution_found = len(unserved_customers) == 0
        
        return {
            'solution_found': solution_found,
            'routes': routes,
            'route_metrics': route_metrics,
            'unserved': unserved_customers,
            'suggestions': suggestions,
            'skipped_customers': skipped_node_indices,
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

    def _extract_solution(self, routing: Any, manager: Any,
                         solution: Any, data: Dict[str, Any], 
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
        
        # Track coordinates that couldn't be routed (for UI warning)
        all_skipped_coords = []
        
        # Track node indices that were skipped during diagnostic (for UI table)
        all_skipped_node_indices = []
        
        # Determine truck type based on vehicle index
        num_small_trucks = len(data.get('small_truck_vehicle_indices', []))
        
        # Extract routes for each vehicle
        for vehicle_id in range(data.get('vehicle_limit', data['num_vehicles'])):
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
                directions_data = self.ors_handler.get_directions(route_coords)

                if directions_data is not None:
                    # Successful ORS response - use detailed geometry and metrics
                    geometry = directions_data['geometry']
                    distance = directions_data['distance']
                    duration = directions_data['duration']
                    
                    # Collect any skipped coordinates (unroutable points)
                    skipped_coords = directions_data.get('skipped_coordinates', [])
                    if skipped_coords:
                        all_skipped_coords.extend(skipped_coords)

                    # Add service time for each customer stop
                    service_time = params.get('service_time_seconds', 300)
                    duration += len(route_nodes) * service_time

                    # Format polylines for frontend (list of coordinate lists)
                    polylines = [geometry] if geometry else []

                else:
                    # Point-by-Point Diagnostic Fallback: Test each segment individually
                    logger.warning(f"ORS Directions API failed for Vehicle {vehicle_id}, attempting point-by-point diagnostic")
                    
                    # Initialize tracking for segment-by-segment testing
                    successful_segments = []  # List of (geometry, distance, duration) tuples
                    skipped_node_indices = []  # Node indices that failed
                    total_distance = 0.0
                    total_duration = 0.0
                    service_time = params.get('service_time_seconds', 300)
                    
                    # Test each consecutive segment in the route
                    for i in range(len(full_route) - 1):
                        from_node = full_route[i]
                        to_node = full_route[i + 1]
                        from_coord = coords[from_node]
                        to_coord = coords[to_node]
                        
                        # Test this segment (two coordinates)
                        segment_coords = [from_coord, to_coord]
                        segment_directions = self.ors_handler.get_directions(segment_coords)
                        
                        if segment_directions is not None:
                            # Segment succeeded - add to successful segments
                            seg_geometry = segment_directions.get('geometry', [])
                            seg_distance = segment_directions.get('distance', 0.0)
                            seg_duration = segment_directions.get('duration', 0.0)
                            
                            # Collect any skipped coordinates from this segment
                            seg_skipped = segment_directions.get('skipped_coordinates', [])
                            if seg_skipped:
                                all_skipped_coords.extend(seg_skipped)
                            
                            successful_segments.append({
                                'geometry': seg_geometry,
                                'distance': seg_distance,
                                'duration': seg_duration
                            })
                            total_distance += seg_distance
                            total_duration += seg_duration
                            
                            logger.debug(f"Vehicle {vehicle_id}: Segment {i}->{i+1} (nodes {from_node}->{to_node}) succeeded")
                        else:
                            # Segment failed - identify the problematic coordinate (usually the destination)
                            # The destination coordinate (to_node) is typically the one that fails
                            if to_node != data['depot']:  # Don't mark depot as skipped
                                if to_node not in skipped_node_indices:
                                    skipped_node_indices.append(to_node)
                                    # Also add to global list for final skipped_customers
                                    if to_node not in all_skipped_node_indices:
                                        all_skipped_node_indices.append(to_node)
                                    skipped_coord = coords[to_node]
                                    all_skipped_coords.append(skipped_coord)
                                    logger.warning(f"Vehicle {vehicle_id}: Segment {i}->{i+1} failed, skipping unroutable node {to_node} (coordinate {skipped_coord})")
                            
                            # Also check if the source coordinate might be the issue (less common)
                            # If this is the first segment and it fails, the source might be problematic
                            if i == 0 and from_node != data['depot']:
                                if from_node not in skipped_node_indices:
                                    # Test if source coordinate alone is problematic
                                    # This is a heuristic - if first segment fails, source might be bad
                                    skipped_node_indices.append(from_node)
                                    if from_node not in all_skipped_node_indices:
                                        all_skipped_node_indices.append(from_node)
                                    skipped_coord = coords[from_node]
                                    all_skipped_coords.append(skipped_coord)
                                    logger.warning(f"Vehicle {vehicle_id}: First segment failed, skipping unroutable source node {from_node} (coordinate {skipped_coord})")
                    
                    # Aggregate successful segments into combined geometry
                    if successful_segments:
                        # Combine all successful segment geometries
                        combined_geometry = []
                        for seg in successful_segments:
                            seg_geom = seg['geometry']
                            if seg_geom:
                                # Skip first point of subsequent segments to avoid duplicates
                                if combined_geometry and len(seg_geom) > 1:
                                    combined_geometry.extend(seg_geom[1:])
                                else:
                                    combined_geometry.extend(seg_geom)
                        
                        # Use combined geometry and metrics
                        geometry = combined_geometry if combined_geometry else [(coords[node][0], coords[node][1]) for node in full_route]
                        distance = total_distance
                        duration = total_duration + (len(route_nodes) - len(skipped_node_indices)) * service_time
                        polylines = [geometry] if geometry else []
                        
                        logger.info(f"Vehicle {vehicle_id}: Successfully aggregated {len(successful_segments)}/{len(full_route)-1} segments. "
                                  f"Skipped {len(skipped_node_indices)} unroutable nodes.")
                    else:
                        # All segments failed - fall back to spider-web
                        logger.warning(f"Vehicle {vehicle_id}: All segments failed. Using Spider Web fallback. "
                                     f"This may indicate rate limiting (429) or all coordinates are unroutable.")
                        
                        # Create straight-line geometry connecting stops in order
                        geometry = [(coords[node][0], coords[node][1]) for node in full_route]  # (lat, lng) tuples
                        
                        # Calculate metrics from internal matrices
                        total_distance = 0.0
                        total_duration = 0.0
                        for i in range(len(full_route) - 1):
                            from_node = full_route[i]
                            to_node = full_route[i + 1]
                            total_distance += data['distance_matrix'][from_node][to_node]
                            total_duration += data['time_matrix'][from_node][to_node]
                        
                        # Add service time for each customer stop
                        total_duration += len(route_nodes) * service_time
                        
                        # Use calculated values
                        distance = total_distance
                        duration = total_duration
                        
                        # Format polylines for frontend (list of coordinate lists)
                        polylines = [geometry]

                # Store route as list (for backward compatibility) and add vehicle_type to metrics
                routes.append(full_route)
                route_metrics.append({
                    'distance': distance,
                    'duration': duration,
                    'polylines': polylines,
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
        
        # Map skipped coordinates to node indices for UI display
        # Start with directly tracked node indices from diagnostic
        skipped_node_indices = list(all_skipped_node_indices)
        
        # Also map coordinates to node indices (for any coordinates added via other paths)
        for skipped_coord in all_skipped_coords:
            # Find the node index that matches this coordinate (approximate match due to float precision)
            for node_idx, node_coord in enumerate(coords):
                if (abs(node_coord[0] - skipped_coord[0]) < 0.0001 and 
                    abs(node_coord[1] - skipped_coord[1]) < 0.0001):
                    if node_idx != 0 and node_idx not in skipped_node_indices:  # Skip depot and avoid duplicates
                        skipped_node_indices.append(node_idx)
                    break
        
        # Log skipped customers
        if skipped_node_indices:
            logger.warning(f"Skipped {len(skipped_node_indices)} unroutable customers during directions fetch")

        return {
            'solution_found': True,
            'routes': routes,
            'route_metrics': route_metrics,
            'unserved': unserved_customers,
            'suggestions': suggestions,
            'skipped_customers': skipped_node_indices,  # Node indices of customers with unroutable coordinates
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

