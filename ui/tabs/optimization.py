"""
Optimization tab for the Warehouse Optimizer Streamlit app.
Handles route optimization and fleet capacity calculations.
"""
import streamlit as st
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional
from config import Config
from calculations.route_optimizer import RouteOptimizer
from calculations.simulation import create_node_map, run_historical_simulation
from exceptions import DataValidationError, GeocodingError, APIRateLimitError

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

    # Retrieve safety buffer from global fleet settings
    safety_buffer = services['fleet_settings'].get('safety_buffer', 1.0)
    
    # Recalculate customer force if raw data is available
    if hasattr(st.session_state, 'raw_data') and st.session_state.raw_data is not None:
        if st.session_state.data is not None:
            try:
                # Recalculate force metrics with the new safety buffer
                updated_data = services['data_manager'].recalculate_customer_force(
                    st.session_state.data,
                    st.session_state.raw_data,
                    buffer=safety_buffer
                )
                st.session_state.data = updated_data
            except Exception as e:
                st.error(f"Error recalculating customer force with safety buffer: {str(e)}")
                logger.error(f"Error recalculating customer force with safety buffer: {e}")

    geocoded_data = st.session_state.data
    valid_coords = geocoded_data.dropna(subset=['lat', 'lng'])
    if valid_coords.empty:
        st.error("No geocoded data.")
        return

    demand_col = 'force_volume'  # Use Customer Force (mean + std × buffer) for daily demand estimate

    if demand_col not in valid_coords.columns:
        st.error(f"Demand column '{demand_col}' not found in the uploaded data.")
        return

    # Use FleetService to get truck capacities (raw volumes in m³)
    capacities = services['fleet_service'].calculate_capacities(
        services['fleet_settings']
    )
    small_capacity_raw = capacities['small_capacity']
    big_capacity_raw = capacities['big_capacity']
    
    # Get safety factor from fleet settings
    safety_factor = services['fleet_settings']['safety_factor']
    
    # Apply safety factor to loads (divide by safety_factor to make loads appear larger)
    # This creates the required buffer when filling the truck
    force_volume = valid_coords[demand_col].fillna(0)
    effective_volume = force_volume / safety_factor
    valid_coords['effective_volume'] = effective_volume
    
    # Scaling factor for OR-Tools (requires integer inputs)
    SCALING_FACTOR = 1000
    
    # Scale effective volumes to integers for OR-Tools
    estimated_demand = np.ceil(effective_volume * SCALING_FACTOR).astype(int)
    
    # Scale capacities for OR-Tools
    small_capacity = int(small_capacity_raw * SCALING_FACTOR)
    big_capacity = int(big_capacity_raw * SCALING_FACTOR)
    
    logger.info(f"Volume-based optimization: safety_factor={safety_factor:.2f}, "
                f"small_capacity={small_capacity_raw:.1f}m³, big_capacity={big_capacity_raw:.1f}m³, "
                f"scaling_factor={SCALING_FACTOR}")

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

            with st.spinner("calculating optimization..."):
                matrix_coords = [wh_coords] + list(zip(valid_coords['lat'], valid_coords['lng']))
                demands = [0] + estimated_demand.tolist()

                # Prepare customer_details for self-healing mechanism
                # Index 0 is warehouse (None), then customer details
                customer_details = [None]  # Warehouse has no customer details
                for idx, row in valid_coords.iterrows():
                    customer_details.append({
                        'name': row.get('שם לקוח', f'Customer {idx}'),
                        'id': str(row.get("מס' לקוח", idx)),
                        'address': row.get('כתובת', row.get('Address', 'Unknown'))
                    })

                # Use strict 1x constraints (no tolerances)
                base_shift_seconds = int(round(services['max_shift_hours'] * 3600))

                logger.info(f"Strict optimization: using raw capacities and time limits")

                params = {
                    'fleet_size': total_fleet_size,
                    'num_small_trucks': num_small_trucks,
                    'num_big_trucks': num_big_trucks,
                    'small_capacity': small_capacity,          # Use raw capacity (1x)
                    'big_capacity': big_capacity,              # Use raw capacity (1x)
                    'max_shift_seconds': base_shift_seconds,   # Use raw time (1x)
                    'service_time_seconds': services['service_time_minutes'] * 60
                }

                # CALL OPTIMIZER
                solution = services['route_optimizer'].solve(
                    [f"Loc {i}" for i in range(len(matrix_coords))],
                    demands,
                    params,
                    coords=matrix_coords,
                    customer_details=customer_details
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
                    
                    
                    # Automatic Simulation: Run historical simulation immediately after optimization
                    required_session_keys = ['raw_data', 'distance_matrix', 'time_matrix', 'node_map']
                    missing_keys = [key for key in required_session_keys if not hasattr(st.session_state, key) or getattr(st.session_state, key) is None]
                    
                    if not missing_keys:
                        try:
                            with st.spinner("Running historical simulation..."):
                                # Build route capacities list based on vehicle type
                                route_metrics = solution.get('route_metrics', [])
                                route_capacities = []
                                small_truck_vol = services['fleet_settings']['small_truck_vol']
                                big_truck_vol = services['fleet_settings']['big_truck_vol']
                                for idx in range(len(solution['routes'])):
                                    if idx < len(route_metrics):
                                        vehicle_type = route_metrics[idx].get('vehicle_type', 'Unknown')
                                        if vehicle_type == 'Small':
                                            route_capacities.append(small_truck_vol)
                                        else:
                                            route_capacities.append(big_truck_vol)
                                    else:
                                        # Default to big truck if unknown
                                        route_capacities.append(big_truck_vol)
                                
                                sim_results = run_historical_simulation(
                                    historical_data=st.session_state.raw_data,
                                    master_routes=solution['routes'],
                                    distance_matrix=st.session_state.distance_matrix,
                                    time_matrix=st.session_state.time_matrix,
                                    node_map=st.session_state.node_map,
                                    service_time_seconds=params['service_time_seconds'],
                                    date_col='תאריך אספקה',
                                    customer_id_col="מס' לקוח",
                                    quantity_col='total_volume_m3',
                                    route_capacities=route_capacities,
                                    max_shift_seconds=params.get('max_shift_seconds')
                                )
                                st.session_state.simulation_results = sim_results
                        except Exception as e:
                            logger.error(f"Simulation failed: {e}")
                            st.warning(f"Routes generated successfully, but historical simulation failed. Error: {str(e)}")
                            st.session_state.simulation_results = None
                else:
                    st.error("Optimization failed.")

        except DataValidationError as e:
            st.error(f"Data validation error: {str(e)}")
            st.info("💡 Please check your Excel file format and ensure all required columns are present.")
            logger.error(f"Data validation error: {e}")
        except GeocodingError as e:
            st.error(f"Geocoding failed: {str(e)}")
            st.info("💡 Please check your warehouse address or ensure you have a valid API key configured.")
            logger.error(f"Geocoding error: {e}")
        except APIRateLimitError as e:
            st.warning(f"API rate limit exceeded: {str(e)}")
            st.info("⏳ Please wait a moment before trying again. The system will retry automatically.")
            logger.warning(f"API rate limit error: {e}")
        except Exception as e:
            st.error(f"Unexpected optimization error: {str(e)}")
            st.info("💡 If this persists, please check your data and try again. Contact support if the issue continues.")
            logger.error(f"Unexpected optimization error: {e}")

    # Persistent Display Block - Show only success/failure status
    if st.session_state.get('solution') is not None:
        solution = st.session_state.solution

        if solution['solution_found']:
            num_routes = len(solution['routes'])
            unserved_list = solution.get('unserved', [])

            if unserved_list:
                st.warning(f"Optimization complete with warnings: {num_routes} trucks used. {len(unserved_list)} customers could not be served.")
            else:
                st.success(f"Optimization complete: {num_routes} trucks used. All customers served.")

            if services.get('map_builder') is None:
                from visualizer import MapBuilder
                services['map_builder'] = MapBuilder()

            wh_coords = st.session_state.warehouse_coords
            if wh_coords:
                try:
                    phase2_map = services['map_builder'].create_phase2_map(
                        solution,
                        valid_coords,
                        wh_coords,
                        geo_service=st.session_state.geo_service
                    )
                    st.components.v1.html(phase2_map._repr_html_(), height=600)
                except Exception as e:
                    st.error(f"Error displaying solution map: {str(e)}")
                    logger.error(f"Error displaying solution map: {e}")
            else:
                st.warning("Warehouse coordinates are missing; map display is unavailable.")

            st.subheader("Route Details")
            route_metrics = solution.get('route_metrics', [])
            for route_idx, route in enumerate(solution['routes']):
                if len(route) <= 2:
                    continue

                metrics = route_metrics[route_idx] if route_idx < len(route_metrics) else {}
                vehicle_type = metrics.get('vehicle_type', 'Unknown')
                with st.expander(f"Route {route_idx + 1} - {vehicle_type} Truck"):
                    if metrics:
                        st.write(f"Distance: {metrics.get('distance', 0):.1f} meters")
                        st.write(f"Duration: {metrics.get('duration', 0) / 3600:.1f} hours")

                    route_customers = []
                    for node_idx in route[1:-1]:
                        if node_idx > 0 and node_idx - 1 < len(valid_coords):
                            customer = valid_coords.iloc[node_idx - 1]
                            route_customers.append({
                                'Customer ID': str(customer.get("מס' לקוח", "")),
                                'Customer Name': customer.get('שם לקוח', 'N/A'),
                                'Address': customer.get('כתובת', 'N/A'),
                                'Customer Force (m³)': round(customer.get('force_volume', 0), 2)
                            })

                    if route_customers:
                        st.dataframe(pd.DataFrame(route_customers), width="stretch")
                    else:
                        st.caption("No customers listed for this route.")
        else:
            st.error("Optimization failed.")
