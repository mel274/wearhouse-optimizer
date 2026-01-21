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

    demand_col = 'force_volume'  # Volume-based demand (Mean + Std*Buffer)

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

    # --- Advanced Solver Controls ---
    with st.sidebar.expander("Advanced Optimization"):
        solver_time_limit_minutes = st.number_input(
            "Solver Time Limit (Minutes)",
            min_value=1,
            value=3,
            help="Maximum time the solver will run. Longer times may yield better solutions for complex problems."
        )
        first_solution_strategy = st.selectbox(
            "Optimization Strategy",
            options=["Global Best", "Constraint Focused"],
            index=0,
            help="Strategy for finding the initial solution. 'Constraint Focused' prioritizes constraints, 'Global Best' seeks globally optimal arcs."
        )
        target_failure_rate = st.slider(
            "Allowed Failure Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=Config.DEFAULT_TARGET_FAILURE_RATE,
            step=0.1,
            help="Maximum acceptable failure rate for routes (volume/time constraints exceeded)."
        )
    
    # Convert minutes to seconds for the solver
    solver_time_limit_seconds = solver_time_limit_minutes * 60

    if st.button("Run Optimization", type="primary"):
        try:
            # Initialize solution variable
            best_solution = None

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

            with st.status("Running Optimization with Route Merging...", expanded=True) as status:
                # Phase 1: Build Master Routes (Initial Clustering)
                status.write("**Phase 1: Building Master Routes (Initial Clustering)**")

                # Preparation Phase
                matrix_coords = [wh_coords] + list(zip(valid_coords['lat'], valid_coords['lng']))
                demands = [0] + estimated_demand.tolist()

                # Use strict capacity limits (1x, no Pressure Cooker buffer)
                opt_small_capacity = small_capacity
                opt_big_capacity = big_capacity

                # Use exact shift time (no tolerance buffers)
                base_shift_seconds = int(round(services['max_shift_hours'] * 3600))
                opt_shift_seconds = base_shift_seconds

                logger.info("Optimizing with strict physical limits for master routes")

                params = {
                    'fleet_size': total_fleet_size,
                    'num_small_trucks': num_small_trucks,
                    'num_big_trucks': num_big_trucks,
                    'small_capacity': opt_small_capacity,      # Strict physical capacity (1x)
                    'big_capacity': opt_big_capacity,          # Strict physical capacity (1x)
                    'max_shift_seconds': opt_shift_seconds,    # Strict physical time limit
                    'service_time_seconds': services['service_time_minutes'] * 60,
                    # Advanced solver controls
                    'time_limit_seconds': solver_time_limit_seconds,
                    'first_solution_strategy': first_solution_strategy,
                    'target_failure_rate': target_failure_rate
                }

                # Solver Phase - Create master routes
                solution = services['route_optimizer'].solve(
                    [f"Loc {i}" for i in range(len(matrix_coords))],
                    demands,
                    params,
                    coords=matrix_coords
                )

                # Safety Guard: Check if solver found a solution
                if solution is not None and solution.get('solution_found', False):
                    best_solution = solution
                    status.write(f"Master routes created: {len(solution['routes'])} routes")

                # Phase 2: Historical Simulation with Route Merging
                sim_results = None
                if solution['solution_found']:
                    status.write("**Phase 2: Daily Simulation with Route Merging**")
                    
                    # Store matrices and node map for historical simulation
                    if 'matrix_data' in solution:
                        matrix_data = solution['matrix_data']
                        st.session_state.distance_matrix = matrix_data['distances']
                        st.session_state.time_matrix = matrix_data['durations']
                    else:
                        logger.warning("Matrix data not found in solution, fetching from API")
                        matrix_data = services['route_optimizer'].ors_handler.get_distance_matrix(matrix_coords)
                        st.session_state.distance_matrix = matrix_data['distances']
                        st.session_state.time_matrix = matrix_data['durations']

                    st.session_state.node_map = create_node_map(valid_coords)
                    st.session_state.optimization_params = params
                    st.session_state.valid_coords_for_simulation = valid_coords

                    # Run simulation with route merging enabled
                    try:
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
                                route_capacities.append(big_truck_vol)

                        # Run simulation WITH route merging enabled
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
                            max_shift_seconds=params.get('max_shift_seconds'),
                            enable_merging=Config.ENABLE_ROUTE_MERGING  # Route merging for daily optimization
                        )
                        
                        status.write(f"Simulation complete: {len(sim_results)} days processed")
                        
                    except Exception as e:
                        logger.error(f"Simulation failed: {e}")
                        sim_results = None

                # Calculate success rate
                if sim_results is not None and not sim_results.empty:
                    total_days = len(sim_results)
                    failed_days_count = 0

                    for _, day_row in sim_results.iterrows():
                        routes = day_row.get('Route_Breakdown', [])
                        if any(not r.get('success', True) for r in routes):
                            failed_days_count += 1

                    success_rate = ((total_days - failed_days_count) / total_days * 100) if total_days > 0 else 0.0
                    
                    # Calculate merge statistics
                    avg_routes_before = sim_results['Routes_Before_Merge'].mean()
                    avg_routes_after = sim_results['Routes_After_Merge'].mean()
                    merge_reduction = (1 - avg_routes_after / avg_routes_before) * 100 if avg_routes_before > 0 else 0
                    
                    status.write(f"Success Rate: {success_rate:.1f}%")
                    status.write(f"Route Merging: Avg {avg_routes_before:.1f} -> {avg_routes_after:.1f} trucks ({merge_reduction:.1f}% reduction)")
                    
                    status.update(label="Optimization Complete!", state="complete", expanded=False)
                    st.toast("Optimization with route merging complete!")
                else:
                    status.update(label="Simulation failed", state="error", expanded=False)
                    
                if not solution['solution_found']:
                    st.error("Solver failed to find a feasible solution. Try increasing fleet size or Time Limits.")
                    best_solution = None

            # Final Persistence: Ensure results are properly stored and UI refreshes
            if best_solution is not None:
                # 1. Persist the Best Result to State
                st.session_state.solution = best_solution
                st.session_state.simulation_results = sim_results

                # 2. Persist other necessary context if needed
                # st.session_state.distance_matrix = ... (if changed)

                # 3. Success Message
                st.toast("Optimization cycle complete. Updating UI...")

                # 4. Force UI Refresh
                st.rerun()

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

            # Aggregate Summary Statistics
            st.subheader("Optimization Results - Aggregate Summary")

            # Check if simulation results exist and are not empty
            if st.session_state.get('simulation_results') is not None and not st.session_state.simulation_results.empty:
                sim_results = st.session_state.simulation_results
                
                # Calculate aggregate statistics
                total_days = len(sim_results)
                
                # Truck usage statistics (with merging)
                avg_trucks_used = sim_results['Routes_After_Merge'].mean() if 'Routes_After_Merge' in sim_results.columns else sim_results['Active_Routes'].mean()
                max_trucks_used = sim_results['Routes_After_Merge'].max() if 'Routes_After_Merge' in sim_results.columns else sim_results['Active_Routes'].max()
                min_trucks_used = sim_results['Routes_After_Merge'].min() if 'Routes_After_Merge' in sim_results.columns else sim_results['Active_Routes'].min()
                
                # Merge reduction statistics
                if 'Routes_Before_Merge' in sim_results.columns and 'Routes_After_Merge' in sim_results.columns:
                    avg_before_merge = sim_results['Routes_Before_Merge'].mean()
                    avg_after_merge = sim_results['Routes_After_Merge'].mean()
                    merge_reduction_pct = (1 - avg_after_merge / avg_before_merge) * 100 if avg_before_merge > 0 else 0
                else:
                    avg_before_merge = avg_trucks_used
                    avg_after_merge = avg_trucks_used
                    merge_reduction_pct = 0
                
                # Distance and time statistics
                avg_daily_distance = sim_results['Total_Distance_km'].mean()
                total_distance = sim_results['Total_Distance_km'].sum()
                avg_daily_time = sim_results['Daily_Total_Duration'].mean() / 3600.0  # Convert to hours
                max_shift_time = sim_results['Max_Shift_Duration_hours'].mean()
                
                # Volume and utilization statistics
                total_load = sim_results['Daily_Total_Load'].sum()
                total_capacity = sim_results['Daily_Total_Capacity'].sum()
                avg_volume_utilization = (total_load / total_capacity * 100) if total_capacity > 0 else 0
                
                # Success rate calculation
                failed_days = 0
                for _, day_row in sim_results.iterrows():
                    routes = day_row.get('Route_Breakdown', [])
                    if any(not r.get('success', True) for r in routes):
                        failed_days += 1
                success_rate = ((total_days - failed_days) / total_days * 100) if total_days > 0 else 0
                
                # Customer statistics
                avg_customers = sim_results['Active_Customers'].mean()
                avg_stops = sim_results['Total_Stops'].mean()
                
                # Display aggregate metrics in a clean layout
                st.markdown("### Fleet Efficiency")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Trucks/Day", f"{avg_trucks_used:.1f}", 
                           delta=f"-{merge_reduction_pct:.0f}% from merging" if merge_reduction_pct > 0 else None,
                           delta_color="normal")
                col2.metric("Truck Range", f"{min_trucks_used:.0f} - {max_trucks_used:.0f}")
                col3.metric("Success Rate", f"{success_rate:.1f}%")
                col4.metric("Days Simulated", f"{total_days}")
                
                st.markdown("### Distance & Time")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Daily Distance", f"{avg_daily_distance:.1f} km")
                col2.metric("Total Distance", f"{total_distance:.0f} km")
                col3.metric("Avg Daily Hours", f"{avg_daily_time:.1f} h")
                col4.metric("Avg Max Shift", f"{max_shift_time:.1f} h")
                
                st.markdown("### Volume & Utilization")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Volume Utilization", f"{avg_volume_utilization:.1f}%")
                col2.metric("Avg Customers/Day", f"{avg_customers:.1f}")
                col3.metric("Avg Stops/Day", f"{avg_stops:.1f}")
                col4.metric("Total Volume Delivered", f"{total_load:.0f} m³")
                
                # Route Merging Summary
                if merge_reduction_pct > 0:
                    st.markdown("### Route Merging Summary")
                    st.info(f"Route merging reduced truck usage from avg **{avg_before_merge:.1f}** to **{avg_after_merge:.1f}** trucks per day (**{merge_reduction_pct:.1f}%** reduction)")
                
            else:
                # Edge case: simulation results not available
                st.info("Simulation statistics are unavailable. Please run optimization to see aggregate results.")

        else:
            st.error("Optimization failed.")
