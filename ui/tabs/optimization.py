"""
Optimization tab for the Warehouse Optimizer Streamlit app.
Handles route optimization and fleet capacity calculations.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from config import Config
from calculations.route_optimizer import RouteOptimizer
from calculations.simulation import create_node_map
from visualizer import MapBuilder
from exceptions import DataValidationError, GeocodingError, APIRateLimitError
import shared.utils as utils


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

    # Initialize route optimizer if not already done
    if services['route_optimizer'] is None:
        services['route_optimizer'] = RouteOptimizer(Config.OPENROUTESERVICE_API_KEY)

    geocoded_data = st.session_state.data
    valid_coords = geocoded_data.dropna(subset=['lat', 'lng'])
    if valid_coords.empty:
        st.error("No geocoded data.")
        return

    demand_col = 'avg_quantity'

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

    # Define fleet composition (default: 6 Small, 12 Big)
    num_small_trucks = 6
    num_big_trucks = 12
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
                    'max_shift_seconds': services['max_shift_hours'] * 3600,
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
                # This will hit the cache (matrices were fetched during optimization)
                matrix_data = services['route_optimizer'].ors_handler.get_distance_matrix(matrix_coords)

                st.session_state.distance_matrix = matrix_data['distances']
                st.session_state.time_matrix = matrix_data['durations']
                st.session_state.node_map = create_node_map(valid_coords)
                st.session_state.optimization_params = params
                st.session_state.valid_coords_for_simulation = valid_coords

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

                    # Expandable Route Details
                    st.subheader("Route Details")

                    route_metrics = solution.get('route_metrics', [])

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

                        warnings = []
                        if current_load > vehicle_capacity: warnings.append("⚠️ Overloaded")
                        if r_metrics.get('duration', 0) > params['max_shift_seconds']: warnings.append("⚠️ Overtime")
                        status_str = " ".join(warnings) if warnings else "✅ OK"

                        header = f"Truck {idx+1} ({vehicle_type}): {len(route)-2} Stops | Load: {current_load} | Dist: {dist_km:.1f} km | Time: {dur_str} {status_str}"

                        with st.expander(header):
                            stops_data = []
                            for stop_order, node_idx in enumerate(route[1:-1]):
                                if node_idx > 0:
                                    customer = valid_coords.iloc[node_idx - 1]
                                    stops_data.append({
                                        "Stop #": stop_order + 1,
                                        "ID": customer.get("מס' לקוח", ""),
                                        "Name": customer.get("שם לקוח", ""),
                                        "Address": customer.get("כתובת", ""),
                                        "Qty": int(customer.get(demand_col, 0))
                                    })

                            if stops_data:
                                st.table(pd.DataFrame(stops_data))
                            else:
                                st.write("Warehouse return only.")

                    # Global Metrics
                    m = solution['metrics']
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Distance", f"{m['total_distance']/1000:.1f} km")
                    c2.metric("Total Time", f"{m['total_time']/3600:.1f} h")
                    c3.metric("Vehicles Used", m['num_vehicles_used'])

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
