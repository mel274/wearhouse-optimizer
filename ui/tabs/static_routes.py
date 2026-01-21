"""
Static Routes tab for the Warehouse Optimizer Streamlit app.
Displays permanent routes with a simplified view (map and route list).
Generates broad static routes using loose constraints.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
from config import Config
from calculations.route_optimizer import RouteOptimizer
from calculations.simulation import create_node_map
from exceptions import DataValidationError, GeocodingError, APIRateLimitError
import shared.utils as utils

logger = logging.getLogger(__name__)


def tab_static_routes(services: Optional[Dict[str, Any]]) -> None:
    """Handle the static routes tab functionality."""
    st.header("Permanent Routes (Broad Clusters)")

    # Handle case where services might be None
    if services is None:
        st.warning("Application configuration error. Please refresh the page.")
        return

    if st.session_state.data is None:
        st.warning("Please upload data first in the Data Upload tab.")
        return

    # Check if API key is available
    if not Config.OPENROUTESERVICE_API_KEY:
        st.warning("Static route generation requires an API Key. Please configure `OPENROUTESERVICE_API_KEY` in your environment variables to use this feature.")
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

    # Use force_quantity or avg_quantity for static routes (not total_quantity)
    demand_col = 'force_quantity'
    if demand_col not in valid_coords.columns:
        # Fallback to avg_quantity if force_quantity is not available
        demand_col = 'avg_quantity'
        if demand_col not in valid_coords.columns:
            st.error(f"Demand columns '{demand_col}' and 'force_quantity' not found in the uploaded data.")
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
    force_quantity = valid_coords[demand_col].fillna(0)
    effective_volume = force_quantity / safety_factor
    valid_coords = valid_coords.copy()
    valid_coords['effective_volume'] = effective_volume

    # Scaling factor for OR-Tools (requires integer inputs)
    SCALING_FACTOR = 1000

    # Scale effective volumes to integers for OR-Tools
    estimated_demand = np.ceil(effective_volume * SCALING_FACTOR).astype(int)

    # Scale capacities for OR-Tools and apply 2x multiplier for loose constraints
    small_capacity = int(small_capacity_raw * SCALING_FACTOR * 2.0)
    big_capacity = int(big_capacity_raw * SCALING_FACTOR * 2.0)

    logger.info(f"Static routes (2x constraints): safety_factor={safety_factor:.2f}, "
                f"small_capacity={small_capacity_raw:.1f}m³-> {small_capacity_raw*2:.1f}m³, big_capacity={big_capacity_raw:.1f}m³-> {big_capacity_raw*2:.1f}m³, "
                f"scaling_factor={SCALING_FACTOR}")

    # Define fleet composition from UI inputs
    num_big_trucks = services.get('num_big_trucks', 12)
    num_small_trucks = services.get('num_small_trucks', 6)
    total_fleet_size = num_small_trucks + num_big_trucks

    if st.button("Generate Static Routes", type="primary"):
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

            with st.spinner("Generating broad static routes..."):
                matrix_coords = [wh_coords] + list(zip(valid_coords['lat'], valid_coords['lng']))
                demands = [0] + estimated_demand.tolist()

                # Apply 2x multiplier to constraints for loose clustering
                base_shift_seconds = int(round(services['max_shift_hours'] * 3600))
                loose_shift_seconds = int(base_shift_seconds * 2.0)

                params = {
                    'fleet_size': total_fleet_size,
                    'num_small_trucks': num_small_trucks,
                    'num_big_trucks': num_big_trucks,
                    'small_capacity': small_capacity,      # 2x capacity
                    'big_capacity': big_capacity,          # 2x capacity
                    'max_shift_seconds': loose_shift_seconds,  # 2x time
                    'service_time_seconds': services['service_time_minutes'] * 60
                }

                # Generate static routes with loose constraints
                solution = services['route_optimizer'].solve(
                    [f"Loc {i}" for i in range(len(matrix_coords))],
                    demands,
                    params,
                    coords=matrix_coords
                )

                st.session_state.static_solution = solution

                # Store matrices and node map for potential simulation
                if 'matrix_data' in solution:
                    matrix_data = solution['matrix_data']
                    st.session_state.static_distance_matrix = matrix_data['distances']
                    st.session_state.static_time_matrix = matrix_data['durations']
                else:
                    logger.warning("Matrix data not found in static solution")
                    matrix_data = services['route_optimizer'].ors_handler.get_distance_matrix(matrix_coords)
                    st.session_state.static_distance_matrix = matrix_data['distances']
                    st.session_state.static_time_matrix = matrix_data['durations']
                st.session_state.static_node_map = create_node_map(valid_coords)
                st.session_state.static_valid_coords = valid_coords

                if solution['solution_found']:
                    st.success(f"Static routes generated successfully: {len(solution['routes'])} broad clusters created.")
                else:
                    st.error("Static route generation failed.")

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
            st.error(f"Unexpected error generating static routes: {str(e)}")
            st.info("💡 If this persists, please check your data and try again. Contact support if the issue continues.")
            logger.error(f"Unexpected static routes error: {e}")

    # Get Data for display
    solution = st.session_state.get('static_solution')
    if solution is None:
        st.info("Click 'Generate Static Routes' to create broad route clusters.")
        return

    valid_coords = st.session_state.get('static_valid_coords')
    wh_coords = st.session_state.warehouse_coords
    if valid_coords is None:
        st.error("Static route data not available.")
        return
    
    # Map Section
    st.header("Permanent Routes Map")
    phase2_map = services['map_builder'].create_phase2_map(
        solution, valid_coords, wh_coords,
        geo_service=st.session_state.geo_service
    )
    st.components.v1.html(phase2_map._repr_html_(), height=700)
    
    # Routes List Section
    st.header("Route Assignments")
    
    route_metrics = solution.get('route_metrics', [])
    
    for idx, route in enumerate(solution['routes']):
        if len(route) <= 2:  # Skip empty routes
            continue

        # Get vehicle type from route metrics
        if idx < len(route_metrics):
            vehicle_type = route_metrics[idx].get('vehicle_type', 'Big')
            metrics = route_metrics[idx]
        else:
            vehicle_type = 'Big'
            metrics = {'distance': 0, 'duration': 0}

        # Calculate detailed metrics for header
        num_stops = len(route) - 2  # Exclude start/end depot
        dist_km = metrics['distance'] / 1000

        # Format time as HH:MM:SS
        duration_seconds = metrics['duration']
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Sum quantities and effective volumes for all customer nodes in route
        total_qty = 0.0
        vol_eff = 0.0
        for node_idx in route[1:-1]:  # Skip depot nodes
            if node_idx > 0:  # Valid customer node
                customer = valid_coords.iloc[node_idx - 1]
                # Use avg_quantity for static routes context
                qty = customer.get('avg_quantity', customer.get('force_quantity', 0))
                total_qty += qty if pd.notna(qty) else 0
                # Use effective_volume for capacity calculations
                eff_vol = customer.get('effective_volume', customer.get('total_volume_m3', 0))
                vol_eff += eff_vol if pd.notna(eff_vol) else 0

        # Get base capacity (not the 2x solver capacity)
        if vehicle_type == 'Small':
            cap_base = services['fleet_settings']['small_truck_vol']
        else:  # Big truck
            cap_base = services['fleet_settings']['big_truck_vol']

        # Calculate percentage usage
        pct_usage = (vol_eff / cap_base) * 100 if cap_base > 0 else 0

        # Determine status
        status_str = "OVERLOAD" if vol_eff > cap_base else "OK"

        # Construct detailed header
        header_label = f"Truck {idx+1} ({vehicle_type}): {num_stops} Stops | Load: {int(total_qty)} | Dist: {dist_km:.1f} km | Time: {time_str} | Vol (Eff): {vol_eff:.1f}/{cap_base} m3 ({pct_usage:.0f}%) {status_str}"

        with st.expander(header_label):
            # Simplified Content: Create list of stops
            stops_data = []
            for stop_order, node_idx in enumerate(route[1:-1]):  # Skip depot (first and last nodes)
                if node_idx > 0:  # Skip depot node
                    customer = valid_coords.iloc[node_idx - 1]
                    stops_data.append({
                        "Stop #": stop_order + 1,
                        "Name": customer.get("שם לקוח", "")
                    })
            
            if stops_data:
                st.dataframe(pd.DataFrame(stops_data), use_container_width=True, hide_index=True)
            else:
                st.write("Warehouse return only.")
