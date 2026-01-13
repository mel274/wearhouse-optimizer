"""
Optimization tab for the Warehouse Optimizer Streamlit app.
Handles route optimization and fleet capacity calculations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from config import Config
from calculations.route_optimizer import RouteOptimizer
from calculations.simulation import create_node_map, run_historical_simulation
from visualizer import MapBuilder
from exceptions import DataValidationError, GeocodingError, APIRateLimitError
import shared.utils as utils

logger = logging.getLogger(__name__)


def tab_optimization(services: Optional[Dict[str, Any]]) -> None:
    """Handle the route optimization tab functionality."""
    st.header("Route Optimization")

    # Handle case where services might be None (shouldn't happen after fix, but defensive)
    if services is None:
        st.warning("Application configuration error. Please refresh the page.")
        return

    if st.session_state.data is None:
        st.warning("Please upload data first in the Data Upload tab.")
        return

    # Check if API key is available for optimization
    if not Config.OPENROUTESERVICE_API_KEY:
        st.warning("Optimization requires an API Key. Please configure `OPENROUTESERVICE_API_KEY` in your environment variables to use this feature.")
        return

    # Initialize services if not already done
    from data_manager import DataManager
    if services['data_manager'] is None:
        services['data_manager'] = DataManager()
    
    # Initialize route optimizer if not already done
    if services['route_optimizer'] is None:
        services['route_optimizer'] = RouteOptimizer(Config.OPENROUTESERVICE_API_KEY)

    # Add Customer Force slider to sidebar
    st.sidebar.markdown("---")
    force_percentile = st.sidebar.slider(
        'customer force percentile',
        min_value=0.5,
        max_value=1.0,
        value=0.8,
        step=0.05
    )

    # Recalculate customer force if raw data is available
    if hasattr(st.session_state, 'raw_data') and st.session_state.raw_data is not None:
        if st.session_state.data is not None:
            try:
                # Recalculate force metrics with the new percentile
                updated_data = services['data_manager'].recalculate_customer_force(
                    st.session_state.data,
                    st.session_state.raw_data,
                    force_percentile
                )
                st.session_state.data = updated_data
            except Exception as e:
                st.error(f"Error recalculating customer force: {str(e)}")
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error recalculating customer force: {e}")

    geocoded_data = st.session_state.data
    valid_coords = geocoded_data.dropna(subset=['lat', 'lng'])
    if valid_coords.empty:
        st.error("No geocoded data.")
        return

    demand_col = 'avg_quantity'  # This contains force_quantity (percentile-based)

    if demand_col not in valid_coords.columns:
        st.error(f"Demand column '{demand_col}' not found in the uploaded data.")
        return

    estimated_demand = np.ceil(valid_coords[demand_col]).astype(int)

    # Use FleetService to calculate capacities
    total_volume = st.session_state.data['total_volume_m3'].sum()
    total_quantity = st.session_state.data['total_quantity'].sum()

    capacities = services['fleet_service'].calculate_capacities(
        total_volume, total_quantity, services['fleet_settings']
    )
    small_capacity = capacities['small_capacity']
    big_capacity = capacities['big_capacity']

    # Define fleet composition from UI inputs
    num_big_trucks = services.get('num_big_trucks', 12)
    num_small_trucks = services.get('num_small_trucks', 6)
    total_fleet_size = num_small_trucks + num_big_trucks

    if st.button("Run Optimization", type="primary"):
        try:
            wh_coords = None
            manual_warehouse_address = services.get('warehouse_address')

            # Prioritize manual address
            if manual_warehouse_address:
                if st.session_state.geo_service is None:
                    st.error("Geocoding service is not available. Please configure an API key to use manual warehouse addresses.")
                    return
                with st.spinner(f"Geocoding manual address: '{manual_warehouse_address}'..."):
                    wh_coords = st.session_state.geo_service.get_coordinates(manual_warehouse_address)
                    if not wh_coords:
                        st.error(f"Could not geocode address: '{manual_warehouse_address}'. Please check the address or try another.")
                        return
            # Fallback to warehouse coordinates from session state (set in Data Upload tab)
            elif st.session_state.warehouse_coords:
                wh_coords = st.session_state.warehouse_coords

            # If no warehouse location is determined, stop.
            if not wh_coords:
                st.error("A warehouse location is required. Please enter an address or enable auto-calculation in the Data Upload tab.")
                return

            st.session_state.warehouse_coords = wh_coords

            with st.spinner("Running Optimization Stages..."):
                matrix_coords = [wh_coords] + list(zip(valid_coords['lat'], valid_coords['lng']))
                demands = [0] + estimated_demand.tolist()

                params = {
                    'fleet_size': total_fleet_size,
                    'num_small_trucks': num_small_trucks,
                    'num_big_trucks': num_big_trucks,
                    'small_capacity': small_capacity,
                    'big_capacity': big_capacity,
                    'max_shift_seconds': int(round(services['max_shift_hours'] * 3600)),
                    'service_time_seconds': services['service_time_minutes'] * 60
                }

                # CALL OPTIMIZER
                solution = services['route_optimizer'].solve(
                    [f"Loc {i}" for i in range(len(matrix_coords))],
                    demands,
                    params,
                    coords=matrix_coords
                )

                st.session_state.solution = solution

                # Store matrices and node map for historical simulation
                # Reuse matrix data from solution to avoid duplicate API calls
                if 'matrix_data' in solution:
                    # Use matrix data already fetched during optimization
                    matrix_data = solution['matrix_data']
                    st.session_state.distance_matrix = matrix_data['distances']
                    st.session_state.time_matrix = matrix_data['durations']
                else:
                    # Fallback: fetch if not in solution (shouldn't happen, but defensive)
                    logger.warning("Matrix data not found in solution, fetching from API")
                    matrix_data = services['route_optimizer'].ors_handler.get_distance_matrix(matrix_coords)
                    st.session_state.distance_matrix = matrix_data['distances']
                    st.session_state.time_matrix = matrix_data['durations']
                st.session_state.node_map = create_node_map(valid_coords)
                st.session_state.optimization_params = params
                st.session_state.valid_coords_for_simulation = valid_coords

                if solution['solution_found']:
                    st.success("Optimization completed successfully!")
                else:
                    st.error("Optimization failed.")
                if solution['solution_found']:
                    st.success("Optimization completed successfully!")
                else:
                    st.error("Optimization failed.")

        except DataValidationError as e:
            st.error(f"Data validation error: {str(e)}")
            st.info("💡 Please check your Excel file format and ensure all required columns are present.")
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Data validation error: {e}")
        except GeocodingError as e:
            st.error(f"Geocoding failed: {str(e)}")
            st.info("💡 Please check your warehouse address or ensure you have a valid API key configured.")
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Geocoding error: {e}")
        except APIRateLimitError as e:
            st.warning(f"API rate limit exceeded: {str(e)}")
            st.info("⏳ Please wait a moment before trying again. The system will retry automatically.")
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"API rate limit error: {e}")
        except Exception as e:
            st.error(f"Unexpected optimization error: {str(e)}")
            st.info("💡 If this persists, please check your data and try again. Contact support if the issue continues.")
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Unexpected optimization error: {e}")

    # Persistent Display Block - Show results if they exist in session state
    if st.session_state.get('solution') is not None:
        solution = st.session_state.solution
        valid_coords = st.session_state.valid_coords_for_simulation
        wh_coords = st.session_state.warehouse_coords

        if solution['solution_found']:
            num_routes = len(solution['routes'])
            unserved_list = solution.get('unserved', [])

            # Status Reporting
            if unserved_list:
                st.warning(f"Optimization complete with warnings: {num_routes} trucks used. {len(unserved_list)} customers could not be served.")
            else:
                st.success(f"Optimization complete: {num_routes} trucks used. All customers served.")

            # MAP Visualization
            st.subheader("Route Map")
            phase2_map = services['map_builder'].create_phase2_map(
                solution, valid_coords, wh_coords,
                geo_service=st.session_state.geo_service
            )
            st.components.v1.html(phase2_map._repr_html_(), height=700)

            # View Mode Toggle
            view_mode = st.radio(
                "Select View Mode",
                options=["Static Route View", "Routes per Day"],
                horizontal=True,
                key="route_view_mode"
            )

            # Expandable Route Details
            st.subheader("Route Details")

            # Retrieve volume limits from fleet settings
            small_truck_vol = services['fleet_settings']['small_truck_vol']
            big_truck_vol = services['fleet_settings']['big_truck_vol']

            route_metrics = solution.get('route_metrics', [])

            if st.session_state.get("route_view_mode") == "Static Route View":
                demands = [0] + np.ceil(valid_coords[demand_col]).astype(int).tolist()
                params = st.session_state.optimization_params

                for idx, route in enumerate(solution['routes']):
                    if len(route) <= 2: continue

                    current_load = sum(demands[node] for node in route)
                    r_metrics = route_metrics[idx] if idx < len(route_metrics) else {'distance': 0, 'duration': 0}
                    dist_km = r_metrics.get('distance', 0) / 1000
                    dur_str = utils.format_time(r_metrics.get('duration', 0))

                    # Get vehicle type from route metrics
                    vehicle_type = r_metrics.get('vehicle_type', 'Unknown')
                    # Get the appropriate capacity based on vehicle type
                    if vehicle_type == 'Small':
                        vehicle_capacity = params.get('small_capacity', small_capacity)
                    else:
                        vehicle_capacity = params.get('big_capacity', big_capacity)

                    # Calculate loaded volume for this route
                    # Sum force_volume (80th percentile) for all customers on this route
                    loaded_volume = 0.0
                    for node_idx in route[1:-1]:  # Skip depot (first and last nodes)
                        if node_idx > 0:  # Skip depot node
                            customer = valid_coords.iloc[node_idx - 1]
                            volume = customer.get('force_volume', 0)
                            if pd.notna(volume):
                                loaded_volume += float(volume)

                    # Determine truck volume capacity based on vehicle type
                    if vehicle_type == 'Small':
                        truck_vol_capacity = small_truck_vol
                    else:
                        truck_vol_capacity = big_truck_vol

                    # Calculate volume utilization percentage
                    vol_percent = (loaded_volume / truck_vol_capacity * 100) if truck_vol_capacity > 0 else 0.0
                    vol_str = f" | Vol: {loaded_volume:.1f} / {truck_vol_capacity:.1f} m³ ({vol_percent:.0f}%)"

                    warnings = []
                    if current_load > vehicle_capacity: warnings.append("⚠️ Overloaded")
                    if r_metrics.get('duration', 0) > params['max_shift_seconds']: warnings.append("⚠️ Overtime")
                    status_str = " ".join(warnings) if warnings else "✅ OK"

                    header = f"Truck {idx+1} ({vehicle_type}): {len(route)-2} Stops | Load: {current_load} | Dist: {dist_km:.1f} km | Time: {dur_str}{vol_str} {status_str}"

                    with st.expander(header):
                        stops_data = []
                        for stop_order, node_idx in enumerate(route[1:-1]):
                            if node_idx > 0:
                                customer = valid_coords.iloc[node_idx - 1]
                                stops_data.append({
                                    "Stop #": stop_order + 1,
                                    "Name": customer.get("שם לקוח", "")
                                })

                        if stops_data:
                            st.table(pd.DataFrame(stops_data))
                        else:
                            st.write("Warehouse return only.")

                # Global Metrics
                m = solution['metrics']

                # Calculate total volume used across all routes
                total_loaded_volume = 0.0
                total_truck_capacity = 0.0

                for idx, route in enumerate(solution['routes']):
                    if len(route) <= 2: continue

                    r_metrics = solution.get('route_metrics', [])
                    if idx < len(r_metrics):
                        vehicle_type = r_metrics[idx].get('vehicle_type', 'Unknown')
                        if vehicle_type == 'Small':
                            truck_capacity = small_truck_vol
                        else:
                            truck_capacity = big_truck_vol

                        # Calculate volume for this route
                        route_volume = 0.0
                        for node_idx in route[1:-1]:
                            if node_idx > 0:
                                customer = valid_coords.iloc[node_idx - 1]
                                volume = customer.get('force_volume', 0)
                                if pd.notna(volume):
                                    route_volume += float(volume)

                        total_loaded_volume += route_volume
                        total_truck_capacity += truck_capacity

                total_vol_percent = (total_loaded_volume / total_truck_capacity * 100) if total_truck_capacity > 0 else 0.0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Distance", f"{m['total_distance']/1000:.1f} km")
                c2.metric("Total Time", f"{m['total_time']/3600:.1f} h")
                c3.metric("Vehicles Used", m['num_vehicles_used'])
                c4.metric("Total Volume Used", f"{total_loaded_volume:.1f} / {total_truck_capacity:.1f} m³ ({total_vol_percent:.0f}%)")

            else:
                # Prerequisites check for Routes per Day view
                required_session_keys = ['raw_data', 'distance_matrix', 'time_matrix', 'node_map']
                missing_keys = [key for key in required_session_keys if not hasattr(st.session_state, key) or getattr(st.session_state, key) is None]

                if missing_keys:
                    st.warning(f"Missing required data for simulation: {', '.join(missing_keys)}. Please run optimization first.")
                    return

                # Run historical simulation
                with st.spinner("Running historical simulation..."):
                    sim_results = run_historical_simulation(
                        historical_data=st.session_state.raw_data,
                        master_routes=solution['routes'],
                        distance_matrix=st.session_state.distance_matrix,
                        time_matrix=st.session_state.time_matrix,
                        node_map=st.session_state.node_map,
                        service_time_seconds=st.session_state.optimization_params['service_time_seconds'],
                        date_col='תאריך אספקה',
                        customer_id_col="מס' לקוח",
                        quantity_col='total_volume_m3'
                    )

                # Data transformation: Pivot from date-centric to route-centric
                route_stats = {}

                # Initialize stats dictionary for each route
                for idx in range(len(solution['routes'])):
                    route_stats[idx] = []

                # Process each simulation day
                for _, row in sim_results.iterrows():
                    date = row['Date']
                    route_breakdown = row.get('Route_Breakdown', [])

                    # Store daily metrics for each route
                    for route_data in route_breakdown:
                        route_id = route_data['route_id']
                        if route_id in route_stats:
                            route_stats[route_id].append({
                                'Date': date,
                                'Stops': route_data['num_stops'],
                                'Volume': route_data['total_load'],
                                'Distance (km)': route_data['distance_meters'] / 1000,  # Convert to km
                                'Time (h)': route_data['duration_seconds'] / 3600     # Convert to hours
                            })

                # Calculate total simulation days
                total_sim_days = len(sim_results)

                # UI Rendering: Per Route Expanders
                for idx, route in enumerate(solution['routes']):
                    if len(route) <= 2:  # Skip empty routes
                        continue

                    route_daily_data = route_stats.get(idx, [])

                    if not route_daily_data:
                        continue

                    # Calculate averages for this route
                    avg_stops = np.mean([day['Stops'] for day in route_daily_data])
                    avg_volume = np.mean([day['Volume'] for day in route_daily_data])
                    avg_distance = np.mean([day['Distance (km)'] for day in route_daily_data])
                    avg_time = np.mean([day['Time (h)'] for day in route_daily_data])

                    # Count days this route was driven
                    days_driven = len(route_daily_data)

                    # Expander header with averages and days driven
                    header = f"Route {idx+1}: Avg {avg_stops:.1f} stops | Avg Vol: {avg_volume:.1f} | Avg Dist: {avg_distance:.1f} km | Avg Time: {avg_time:.1f} h | Driven: {days_driven}/{total_sim_days} days"

                    with st.expander(header):
                        # Create DataFrame for daily stats
                        daily_df = pd.DataFrame(route_daily_data)
                        # Format Date column to clean string format
                        if 'Date' in daily_df.columns:
                            daily_df['Date'] = pd.to_datetime(daily_df['Date']).dt.strftime('%Y-%m-%d')
                        st.dataframe(daily_df, use_container_width=True, hide_index=True)

                # Footer: System-wide daily averages and totals
                if not sim_results.empty:
                    system_avg_distance = sim_results['Total_Distance_km'].mean()
                    system_avg_time = sim_results['Max_Shift_Duration_hours'].mean()

                    # Calculate grand totals for the entire simulation period
                    total_distance_all_days = sim_results['Total_Distance_km'].sum()
                    total_time_all_days = sim_results['Max_Shift_Duration_hours'].sum()

                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("System Avg Daily Distance", f"{system_avg_distance:.1f} km")
                    col2.metric("Total Daily Distance", f"{total_distance_all_days:.1f} km")
                    col3.metric("System Avg Daily Time", f"{system_avg_time:.1f} hours")
                    col4.metric("Total Daily Time", f"{total_time_all_days:.1f} hours")

        else:
            st.error("Optimization failed.")
