"""
Historical backtesting simulation engine.
Simulates how optimized master routes would perform on historical daily data.
Uses cached distance/time matrices - NO API calls.
"""
import logging
import pandas as pd
from typing import List, Dict, Any, Tuple, Set
from datetime import datetime

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
    depot: int = 0
) -> pd.DataFrame:
    """
    Run backtesting simulation on historical data.

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

    Returns:
        DataFrame with daily metrics: Date, Total_Distance_km, Max_Shift_Duration_hours,
        Fleet_Utilization_pct, Active_Customers, Variance_Customers, Route_Breakdown
    """
    logger.info(f"Starting historical simulation for {len(historical_data)} historical records")

    # Parse dates if needed
    if historical_data[date_col].dtype == 'object':
        historical_data[date_col] = pd.to_datetime(historical_data[date_col], dayfirst=True, errors='coerce')

    # Create reverse node map (matrix index -> customer ID) for load calculations
    reverse_node_map = {v: k for k, v in node_map.items()}

    # Group by date
    daily_results = []
    unique_dates = historical_data[date_col].dropna().unique()

    logger.info(f"Processing {len(unique_dates)} unique dates")

    for date in sorted(unique_dates):
        # Get active customers for this date
        day_data = historical_data[historical_data[date_col] == date]
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

        # Calculate metrics for each master route (sub-routes)
        daily_total_distance = 0.0
        daily_max_duration = 0.0
        daily_total_stops = 0
        active_routes = 0
        route_breakdown = []

        for route_idx, master_route in enumerate(master_routes):
            # Create sub-route for this date
            sub_route = create_sub_route(master_route, active_nodes, depot)

            # Only process routes with at least one customer stop
            if len(sub_route) > 2:  # More than just [depot, depot]
                metrics = calculate_route_metrics(
                    sub_route,
                    distance_matrix,
                    time_matrix,
                    service_time_seconds,
                    depot
                )

                # Calculate total load for this route
                total_load = 0.0
                for node in sub_route:
                    if node != depot and node in reverse_node_map:
                        customer_id = reverse_node_map[node]
                        customer_orders = day_data[day_data[customer_id_col] == customer_id]
                        if not customer_orders.empty:
                            # Sum quantity for this customer on this date
                            customer_quantity = customer_orders[quantity_col].sum()
                            if pd.notna(customer_quantity):
                                total_load += float(customer_quantity)

                # Add granular route metrics
                route_breakdown.append({
                    "route_id": route_idx,
                    "num_stops": metrics['num_stops'],
                    "total_load": total_load,
                    "distance_meters": metrics['distance'],
                    "duration_seconds": metrics['duration']
                })

                # Aggregate daily totals
                daily_total_distance += metrics['distance']
                daily_max_duration = max(daily_max_duration, metrics['duration'])
                daily_total_stops += metrics['num_stops']
                active_routes += 1

        # Calculate fleet utilization (percentage of routes used)
        fleet_utilization = (active_routes / len(master_routes) * 100) if master_routes else 0

        # Store daily results
        daily_results.append({
            'Date': date,
            'Total_Distance_km': daily_total_distance / 1000.0,  # Convert meters to km
            'Max_Shift_Duration_hours': daily_max_duration / 3600.0,  # Convert seconds to hours
            'Fleet_Utilization_pct': fleet_utilization,
            'Active_Customers': len(active_nodes),
            'Variance_Customers': len(variance_customers),
            'Total_Stops': daily_total_stops,
            'Active_Routes': active_routes,
            'Route_Breakdown': route_breakdown
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(daily_results)
    
    logger.info(f"Simulation complete: {len(results_df)} days processed")
    return results_df
