"""
Historical backtesting simulation engine.
Simulates how optimized master routes would perform on historical daily data.
Uses cached distance/time matrices - NO API calls.
"""
import logging
import pandas as pd
from typing import List, Dict, Any, Tuple, Set
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def create_node_map(optimization_data: pd.DataFrame, customer_id_col: str = "מס' לקוח") -> Dict[Any, int]:
    """
    Create a mapping from Customer ID to Matrix Index.
    
    Args:
        optimization_data: DataFrame used during optimization (valid_coords)
        customer_id_col: Column name containing customer IDs
        
    Returns:
        Dictionary mapping Customer ID -> Matrix Index
        Note: Matrix index 0 is the warehouse, customers start at index 1
    """
    node_map = {}
    
    # Matrix index 0 is warehouse (no customer ID)
    # Customers start at matrix index 1
    for idx, row in optimization_data.iterrows():
        customer_id = row.get(customer_id_col)
        if pd.notna(customer_id):
            matrix_index = idx + 1  # +1 because warehouse is at index 0
            node_map[customer_id] = matrix_index
    
    logger.info(f"Created node map: {len(node_map)} customers mapped to matrix indices")
    return node_map


def create_sub_route(master_route: List[int], active_nodes: Set[int], depot: int = 0) -> List[int]:
    """
    Create a sub-route from master route containing only active nodes.
    Preserves the original sequence from the master route.
    
    Args:
        master_route: Original route as list of matrix indices [0, 1, 3, 5, 0]
        active_nodes: Set of matrix indices that are active on this date
        depot: Matrix index of depot (usually 0)
        
    Returns:
        Sub-route with only active nodes, preserving sequence
        Always starts and ends with depot: [0, ...active_nodes..., 0]
    """
    if not master_route:
        return [depot, depot]
    
    # Filter route to only include active nodes (and depot)
    sub_route = []
    for node in master_route:
        if node == depot or node in active_nodes:
            sub_route.append(node)
    
    # Ensure route starts and ends with depot
    if not sub_route or sub_route[0] != depot:
        sub_route.insert(0, depot)
    if not sub_route or sub_route[-1] != depot:
        sub_route.append(depot)
    
    # Remove consecutive duplicate depots
    cleaned = [sub_route[0]]
    for i in range(1, len(sub_route)):
        if sub_route[i] != sub_route[i-1]:
            cleaned.append(sub_route[i])
    
    return cleaned if len(cleaned) > 1 else [depot, depot]


def calculate_route_metrics(
    route: List[int],
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
    service_time_seconds: int,
    depot: int = 0
) -> Dict[str, float]:
    """
    Calculate route metrics using cached matrices (NO API calls).
    
    Args:
        route: Route as list of matrix indices [0, 1, 3, 0]
        distance_matrix: Cached distance matrix
        time_matrix: Cached time matrix
        service_time_seconds: Service time per stop in seconds
        depot: Matrix index of depot
        
    Returns:
        Dictionary with 'distance' (meters), 'duration' (seconds), 'num_stops'
    """
    if len(route) <= 2:  # Only depot or empty route
        return {
            'distance': 0.0,
            'duration': 0.0,
            'num_stops': 0
        }
    
    total_distance = 0.0
    total_duration = 0.0
    num_customer_stops = 0
    
    # Sum distances and times along the route
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        
        # Get values from matrices
        if from_node < len(distance_matrix) and to_node < len(distance_matrix[from_node]):
            total_distance += distance_matrix[from_node][to_node]
        if from_node < len(time_matrix) and to_node < len(time_matrix[from_node]):
            total_duration += time_matrix[from_node][to_node]
        
        # Count customer stops (not depot)
        if to_node != depot:
            num_customer_stops += 1
    
    # Add service time for all customer stops
    total_duration += num_customer_stops * service_time_seconds
    
    return {
        'distance': total_distance,
        'duration': total_duration,
        'num_stops': num_customer_stops
    }


def get_daily_demands(
    day_data: pd.DataFrame,
    node_map: Dict[Any, int],
    customer_id_col: str,
    quantity_col: str
) -> Dict[int, float]:
    """
    Extract actual daily demands per matrix node from the day's data.
    
    Args:
        day_data: DataFrame containing orders for a specific date
        node_map: Mapping of Customer ID -> Matrix Index
        customer_id_col: Column name containing customer IDs
        quantity_col: Column name containing quantity/volume data
        
    Returns:
        Dictionary mapping Matrix Index -> Total demand for that day
    """
    daily_demands = {}
    
    for customer_id, group in day_data.groupby(customer_id_col):
        if customer_id in node_map:
            node_idx = node_map[customer_id]
            total_demand = group[quantity_col].sum()
            if pd.notna(total_demand):
                daily_demands[node_idx] = float(total_demand)
    
    return daily_demands


def calculate_route_load(
    route: List[int],
    daily_demands: Dict[int, float],
    depot: int = 0
) -> float:
    """
    Calculate total load for a route based on daily demands.
    
    Args:
        route: Route as list of matrix indices [0, 1, 3, 0]
        daily_demands: Dictionary mapping node index -> demand
        depot: Matrix index of depot
        
    Returns:
        Total load for the route
    """
    total_load = 0.0
    for node in route:
        if node != depot and node in daily_demands:
            total_load += daily_demands[node]
    return total_load


def can_merge_routes(
    route_a: List[int],
    route_b: List[int],
    daily_demands: Dict[int, float],
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
    service_time_seconds: int,
    capacity_limit: float,
    time_limit: int,
    depot: int = 0
) -> Tuple[bool, List[int], Dict[str, float]]:
    """
    Check if two routes can be merged without exceeding limits.
    
    Args:
        route_a: First route [0, 1, 3, 0]
        route_b: Second route [0, 5, 7, 0]
        daily_demands: Dictionary mapping node index -> demand
        distance_matrix: Cached distance matrix
        time_matrix: Cached time matrix
        service_time_seconds: Service time per stop
        capacity_limit: Maximum volume capacity
        time_limit: Maximum shift time in seconds
        depot: Matrix index of depot
        
    Returns:
        Tuple of (can_merge: bool, merged_route: List[int], metrics: Dict)
    """
    # Extract customer nodes from both routes (exclude depot)
    customers_a = [n for n in route_a if n != depot]
    customers_b = [n for n in route_b if n != depot]
    
    # If both routes are empty, nothing to merge
    if not customers_a and not customers_b:
        return False, [], {}
    
    # Calculate combined load
    combined_load = calculate_route_load(route_a, daily_demands, depot) + \
                    calculate_route_load(route_b, daily_demands, depot)
    
    # Check capacity limit
    if capacity_limit is not None and combined_load > capacity_limit:
        return False, [], {}
    
    # Create merged route: depot -> customers_a -> customers_b -> depot
    # This preserves the sequence from each original route
    merged_route = [depot] + customers_a + customers_b + [depot]
    
    # Calculate metrics for merged route
    metrics = calculate_route_metrics(
        merged_route,
        distance_matrix,
        time_matrix,
        service_time_seconds,
        depot
    )
    
    # Check time limit
    if time_limit is not None and metrics['duration'] > time_limit:
        return False, [], {}
    
    # Add load to metrics
    metrics['total_load'] = combined_load
    
    return True, merged_route, metrics


def find_best_merge_pair(
    routes: List[List[int]],
    route_loads: List[float],
    daily_demands: Dict[int, float],
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
    service_time_seconds: int,
    capacity_limit: float,
    time_limit: int,
    depot: int = 0
) -> Tuple[int, int, List[int], Dict[str, float]]:
    """
    Find the best pair of routes to merge (maximizes utilization).
    
    Strategy: Sort routes by load (lowest first), try to merge smallest routes
    into others to maximize truck utilization.
    
    Args:
        routes: List of routes
        route_loads: List of loads for each route
        daily_demands: Dictionary mapping node index -> demand
        distance_matrix: Cached distance matrix
        time_matrix: Cached time matrix
        service_time_seconds: Service time per stop
        capacity_limit: Maximum volume capacity
        time_limit: Maximum shift time in seconds
        depot: Matrix index of depot
        
    Returns:
        Tuple of (idx_a, idx_b, merged_route, metrics) or (-1, -1, [], {}) if no merge possible
    """
    n = len(routes)
    if n < 2:
        return -1, -1, [], {}
    
    # Create list of (index, load) and sort by load (lowest first)
    indexed_loads = [(i, route_loads[i]) for i in range(n) if len(routes[i]) > 2]
    indexed_loads.sort(key=lambda x: x[1])
    
    best_merge = (-1, -1, [], {})
    best_utilization = 0.0
    
    # Try to merge smallest routes into larger ones
    for i in range(len(indexed_loads)):
        idx_a = indexed_loads[i][0]
        route_a = routes[idx_a]
        
        for j in range(i + 1, len(indexed_loads)):
            idx_b = indexed_loads[j][0]
            route_b = routes[idx_b]
            
            can_merge, merged_route, metrics = can_merge_routes(
                route_a, route_b,
                daily_demands,
                distance_matrix,
                time_matrix,
                service_time_seconds,
                capacity_limit,
                time_limit,
                depot
            )
            
            if can_merge:
                # Calculate utilization for this merge
                utilization = metrics['total_load'] / capacity_limit if capacity_limit else 0
                
                # Prefer merges that maximize utilization
                if utilization > best_utilization:
                    best_utilization = utilization
                    best_merge = (idx_a, idx_b, merged_route, metrics)
    
    return best_merge


def merge_daily_routes(
    sub_routes: List[List[int]],
    daily_demands: Dict[int, float],
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
    service_time_seconds: int,
    capacity_limit: float,
    time_limit: int,
    depot: int = 0
) -> Tuple[List[List[int]], List[Dict[str, float]]]:
    """
    Merge underutilized routes to maximize truck efficiency.
    
    Starts with exact master route structure (as sub-routes) and greedily
    merges routes that can be combined without exceeding capacity or time limits.
    
    Args:
        sub_routes: List of sub-routes (filtered master routes for this day)
        daily_demands: Dictionary mapping node index -> demand for this day
        distance_matrix: Cached distance matrix
        time_matrix: Cached time matrix
        service_time_seconds: Service time per stop
        capacity_limit: Maximum volume capacity (strict 1x, no buffer)
        time_limit: Maximum shift time in seconds
        depot: Matrix index of depot
        
    Returns:
        Tuple of (merged_routes, route_metrics_list)
    """
    # Filter out empty routes (only depot -> depot)
    active_routes = []
    for route in sub_routes:
        if len(route) > 2:  # Has at least one customer
            active_routes.append(list(route))  # Make a copy
    
    if not active_routes:
        return [], []
    
    # Calculate initial loads for all routes
    route_loads = [calculate_route_load(r, daily_demands, depot) for r in active_routes]
    
    # Greedy merge loop
    merged = True
    while merged and len(active_routes) > 1:
        merged = False
        
        # Find best merge pair
        idx_a, idx_b, merged_route, metrics = find_best_merge_pair(
            active_routes,
            route_loads,
            daily_demands,
            distance_matrix,
            time_matrix,
            service_time_seconds,
            capacity_limit,
            time_limit,
            depot
        )
        
        if idx_a >= 0 and idx_b >= 0:
            # Perform the merge: replace route_a with merged, remove route_b
            # Remove higher index first to avoid index shift issues
            if idx_a > idx_b:
                idx_a, idx_b = idx_b, idx_a
            
            active_routes[idx_a] = merged_route
            route_loads[idx_a] = metrics['total_load']
            
            del active_routes[idx_b]
            del route_loads[idx_b]
            
            merged = True
            logger.debug(f"Merged routes: {len(active_routes)} routes remaining, utilization: {metrics['total_load']:.1f}")
    
    # Calculate final metrics for all routes
    final_metrics = []
    for route in active_routes:
        metrics = calculate_route_metrics(
            route,
            distance_matrix,
            time_matrix,
            service_time_seconds,
            depot
        )
        metrics['total_load'] = calculate_route_load(route, daily_demands, depot)
        final_metrics.append(metrics)
    
    logger.debug(f"Route merging complete: {len(sub_routes)} -> {len(active_routes)} routes")
    return active_routes, final_metrics


def _process_single_date(
    date: Any,
    day_data: pd.DataFrame,
    master_routes: List[List[int]],
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
    node_map: Dict[Any, int],
    reverse_node_map: Dict[int, Any],
    service_time_seconds: int,
    depot: int,
    route_capacities: List[float],
    max_shift_seconds: int,
    customer_id_col: str,
    quantity_col: str,
    enable_merging: bool = True
) -> Dict[str, Any]:
    """
    Process a single date in the historical simulation with route merging.

    Args:
        date: The date to process
        day_data: Pre-filtered DataFrame containing only data for this date
        master_routes: List of optimized routes
        distance_matrix: Cached distance matrix
        time_matrix: Cached time matrix
        node_map: Mapping of Customer ID -> Matrix Index
        reverse_node_map: Mapping of Matrix Index -> Customer ID
        service_time_seconds: Service time per stop in seconds
        depot: Matrix index of depot
        route_capacities: List of volume limits for each master route
        max_shift_seconds: Global shift duration limit in seconds
        customer_id_col: Column name containing customer IDs
        quantity_col: Column name containing quantity/volume data
        enable_merging: If True, merge underutilized routes to maximize efficiency

    Returns:
        Dictionary with daily metrics for this date
    """
    # Get active customers for this date
    active_customer_ids = set(day_data[customer_id_col].dropna().unique())

    # Convert customer IDs to matrix indices
    active_nodes = set()
    variance_customers = []

    for customer_id in active_customer_ids:
        if customer_id in node_map:
            active_nodes.add(node_map[customer_id])
        else:
            # Customer not in master routes (variance/ghost customer)
            variance_customers.append(customer_id)

    # Step 1: Create sub-routes from master routes (filter to active customers only)
    sub_routes = []
    for master_route in master_routes:
        sub_route = create_sub_route(master_route, active_nodes, depot)
        sub_routes.append(sub_route)
    
    # Step 2: Get actual daily demands for each customer
    daily_demands = get_daily_demands(day_data, node_map, customer_id_col, quantity_col)
    
    # Step 3: Determine capacity limit (use first route's capacity or max)
    capacity_limit = None
    if route_capacities is not None and len(route_capacities) > 0:
        capacity_limit = max(route_capacities)  # Use max capacity for merging
    
    # Step 4: Merge underutilized routes if enabled
    if enable_merging and capacity_limit is not None:
        merged_routes, merged_metrics = merge_daily_routes(
            sub_routes=sub_routes,
            daily_demands=daily_demands,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            service_time_seconds=service_time_seconds,
            capacity_limit=capacity_limit,
            time_limit=max_shift_seconds,
            depot=depot
        )
    else:
        # No merging - use sub-routes as-is
        merged_routes = [r for r in sub_routes if len(r) > 2]
        merged_metrics = []
        for route in merged_routes:
            metrics = calculate_route_metrics(route, distance_matrix, time_matrix, service_time_seconds, depot)
            metrics['total_load'] = calculate_route_load(route, daily_demands, depot)
            merged_metrics.append(metrics)
    
    # Step 5: Calculate daily totals from merged routes
    daily_total_distance = 0.0
    daily_max_duration = 0.0
    daily_total_stops = 0
    daily_total_duration = 0.0
    daily_total_capacity = 0.0
    daily_total_load = 0.0
    route_breakdown = []
    
    for route_idx, (route, metrics) in enumerate(zip(merged_routes, merged_metrics)):
        total_load = metrics.get('total_load', 0.0)
        
        # Add to daily totals
        daily_total_distance += metrics['distance']
        daily_total_duration += metrics['duration']
        daily_max_duration = max(daily_max_duration, metrics['duration'])
        daily_total_stops += metrics['num_stops']
        daily_total_load += total_load
        
        # Use capacity limit for all merged routes
        limit_vol = capacity_limit
        limit_time = max_shift_seconds
        
        if limit_vol is not None:
            daily_total_capacity += limit_vol
        
        # Check success/failure status (strict limits)
        vol_ok = True if limit_vol is None else (total_load <= limit_vol)
        time_ok = True if limit_time is None else (metrics['duration'] <= limit_time)
        is_success = vol_ok and time_ok
        
        # Determine status string
        if is_success:
            status_string = "OK"
        else:
            status_parts = []
            if not vol_ok:
                status_parts.append("Overload")
            if not time_ok:
                status_parts.append("Overtime")
            status_string = "+".join(status_parts)
        
        # Add granular route metrics
        route_data = {
            "route_id": route_idx,
            "num_stops": metrics['num_stops'],
            "total_load": total_load,
            "distance_meters": metrics['distance'],
            "duration_seconds": metrics['duration'],
            "success": is_success,
            "status": status_string
        }
        
        if limit_vol is not None:
            route_data["limit_vol"] = limit_vol
        if limit_time is not None:
            route_data["limit_time"] = limit_time
        
        route_breakdown.append(route_data)
    
    active_routes = len(merged_routes)
    
    # Calculate fleet utilization (percentage of original master routes used after merging)
    fleet_utilization = (active_routes / len(master_routes) * 100) if master_routes else 0

    # Return daily results for this date
    return {
        'Date': date,
        'Total_Distance_km': daily_total_distance / 1000.0,  # Convert meters to km
        'Max_Shift_Duration_hours': daily_max_duration / 3600.0,  # Convert seconds to hours
        'Fleet_Utilization_pct': fleet_utilization,
        'Active_Customers': len(active_nodes),
        'Variance_Customers': len(variance_customers),
        'Total_Stops': daily_total_stops,
        'Active_Routes': active_routes,
        'Route_Breakdown': route_breakdown,
        'Daily_Total_Duration': daily_total_duration,
        'Daily_Total_Capacity': daily_total_capacity,
        'Daily_Total_Load': daily_total_load,
        'Routes_Before_Merge': len([r for r in sub_routes if len(r) > 2]),
        'Routes_After_Merge': active_routes
    }


def run_historical_simulation(
    historical_data: pd.DataFrame,
    master_routes: List[List[int]],
    distance_matrix: List[List[float]],
    time_matrix: List[List[float]],
    node_map: Dict[Any, int],
    service_time_seconds: int,
    date_col: str = 'תאריך אספקה',
    customer_id_col: str = "מס' לקוח",
    quantity_col: str = 'כמות',
    depot: int = 0,
    route_capacities: List[float] = None,
    max_shift_seconds: int = None,
    enable_merging: bool = True
) -> pd.DataFrame:
    """
    Run backtesting simulation on historical data with route merging optimization.

    This function simulates how master routes perform on historical daily data,
    with optional route merging to maximize truck efficiency.

    Args:
        historical_data: DataFrame with historical orders (must have date and customer ID columns)
        master_routes: List of optimized routes (each route is list of matrix indices)
        distance_matrix: Cached distance matrix from optimization
        time_matrix: Cached time matrix from optimization
        node_map: Mapping of Customer ID -> Matrix Index
        service_time_seconds: Service time per stop in seconds
        date_col: Column name containing dates
        customer_id_col: Column name containing customer IDs
        quantity_col: Column name containing quantity/volume data
        depot: Matrix index of depot (usually 0)
        route_capacities: Optional list of volume limits for each master route
        max_shift_seconds: Optional global shift duration limit in seconds
        enable_merging: If True, merge underutilized routes to maximize efficiency (default: True)

    Returns:
        DataFrame with daily metrics including:
        - Total_Distance_km, Max_Shift_Duration_hours, Fleet_Utilization_pct
        - Active_Customers, Variance_Customers, Total_Stops, Active_Routes
        - Route_Breakdown with success status and limits
        - Routes_Before_Merge, Routes_After_Merge (when merging is enabled)
    """
    logger.info(f"Starting historical simulation for {len(historical_data)} historical records (merging={'enabled' if enable_merging else 'disabled'})")

    # Hebrew date column heuristic - find column containing "תאריך" if date_col not found
    if date_col not in historical_data.columns:
        hebrew_date_cols = [col for col in historical_data.columns if 'תאריך' in col]
        if hebrew_date_cols:
            date_col = hebrew_date_cols[0]
            logger.info(f"Using Hebrew date column: '{date_col}'")
        else:
            raise ValueError(f"Date column '{date_col}' not found and no Hebrew date column containing 'תאריך' was found in available columns: {list(historical_data.columns)}")

    # Parse dates if needed
    if historical_data[date_col].dtype == 'object':
        historical_data[date_col] = pd.to_datetime(historical_data[date_col], dayfirst=True, errors='coerce')

    # Create reverse node map (matrix index -> customer ID) for load calculations
    reverse_node_map = {v: k for k, v in node_map.items()}

    # Group by date
    daily_results = []
    unique_dates = historical_data[date_col].dropna().unique()

    logger.info(f"Processing {len(unique_dates)} unique dates")

    # Use ProcessPoolExecutor to run days in parallel
    with ProcessPoolExecutor() as executor:
        futures = []
        for date in sorted(unique_dates):
            # Filter data for this specific date here to minimize pickling size
            day_data = historical_data[historical_data[date_col] == date]
            
            futures.append(executor.submit(
                _process_single_date,
                date=date,
                day_data=day_data,  # Pass pre-filtered data
                master_routes=master_routes,
                distance_matrix=distance_matrix,
                time_matrix=time_matrix,
                node_map=node_map,
                reverse_node_map=reverse_node_map,
                service_time_seconds=service_time_seconds,
                depot=depot,
                route_capacities=route_capacities,
                max_shift_seconds=max_shift_seconds,
                customer_id_col=customer_id_col,
                quantity_col=quantity_col,
                enable_merging=enable_merging
            ))
        
        for future in as_completed(futures):
            try:
                result = future.result()
                daily_results.append(result)
            except Exception as e:
                logger.error(f"Day simulation failed: {e}")
    
    # Sort results by date after collecting (since parallel execution scrambles order)
    daily_results.sort(key=lambda x: x['Date'])
    
    # Create results DataFrame
    results_df = pd.DataFrame(daily_results)
    
    # Log merging statistics if enabled
    if enable_merging and not results_df.empty:
        avg_before = results_df['Routes_Before_Merge'].mean()
        avg_after = results_df['Routes_After_Merge'].mean()
        logger.info(f"Route merging: avg {avg_before:.1f} routes -> {avg_after:.1f} routes ({(1 - avg_after/avg_before)*100:.1f}% reduction)")
    
    logger.info(f"Simulation complete: {len(results_df)} days processed")
    return results_df
