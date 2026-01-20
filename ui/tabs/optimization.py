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
            # Get target from session state (set by the slider in Phase 1)
            target_failure_rate = st.session_state.get('target_failure_rate', Config.DEFAULT_TARGET_FAILURE_RATE)

            # Initialize iterative loop variables
            current_multipliers = {}  # Start empty (defaults will be used by DataManager)
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

            with st.status("Running Iterative Optimization...", expanded=True) as status:
                for iteration in range(1, Config.MAX_OPTIMIZATION_ITERATIONS + 1):
                    status.write(f"**Iteration {iteration}/{Config.MAX_OPTIMIZATION_ITERATIONS}**")

                    # 1. Update Data Phase
                    # Pass current_multipliers to recalculate forces dynamically
                    st.session_state.data = services['data_manager'].recalculate_customer_force(
                        st.session_state.data,
                        st.session_state.raw_data,
                        buffer_multipliers=current_multipliers
                    )

                    # 2. Preparation Phase
                    matrix_coords = [wh_coords] + list(zip(valid_coords['lat'], valid_coords['lng']))
                    demands = [0] + estimated_demand.tolist()

                    # Pressure Cooker: Double capacity for solver to ensure solution (2x)
                    # Simulation will use real capacity (1x) to detect actual failures
                    opt_small_capacity = small_capacity * 2  # 2x for solver
                    opt_big_capacity = big_capacity * 2      # 2x for solver

                    # Use exact shift time (no tolerance buffers)
                    base_shift_seconds = int(round(services['max_shift_hours'] * 3600))
                    opt_shift_seconds = base_shift_seconds

                    logger.info("Optimizing with strict physical limits (no tolerance buffers)")

                    params = {
                        'fleet_size': total_fleet_size,
                        'num_small_trucks': num_small_trucks,
                        'num_big_trucks': num_big_trucks,
                        'small_capacity': opt_small_capacity,      # Strict physical capacity
                        'big_capacity': opt_big_capacity,          # Strict physical capacity
                        'max_shift_seconds': opt_shift_seconds,    # Strict physical time limit
                        'service_time_seconds': services['service_time_minutes'] * 60,
                        # Advanced solver controls
                        'time_limit_seconds': solver_time_limit_seconds,
                        'first_solution_strategy': first_solution_strategy,
                        # Iterative risk-managed engine
                        'target_failure_rate': target_failure_rate
                    }

                    # 3. Solver Phase
                    solution = services['route_optimizer'].solve(
                        [f"Loc {i}" for i in range(len(matrix_coords))],
                        demands,
                        params,
                        coords=matrix_coords
                    )

                    # Safety Guard: Check if solver found a solution
                    if solution is not None and solution.get('solution_found', False):
                        # Temporarily assume latest is best (Phase 2 - will be improved in Phase 3)
                        best_solution = solution

                    # 4. Simulation Phase (The "Reality Check")
                    sim_results = None
                    if solution['solution_found']:
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

                        # Run simulation
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
                        except Exception as e:
                            logger.error(f"Simulation failed in iteration {iteration}: {e}")
                            sim_results = None

                    # 5. Failure Rate Calculation
                    if sim_results is not None and not sim_results.empty:
                        total_days = len(sim_results)
                        failed_days_count = 0

                        for _, day_row in sim_results.iterrows():
                            routes = day_row.get('Route_Breakdown', [])
                            # Check if any route on this day failed
                            if any(not r.get('success', True) for r in routes):
                                failed_days_count += 1

                        global_failure_rate = (failed_days_count / total_days * 100) if total_days > 0 else 0.0
                    else:
                        global_failure_rate = 100.0  # Default to fail if no sim data

                    status.write(f"Result: {global_failure_rate:.2f}% Failure Rate (Target: {target_failure_rate}%)")

                    # 6. Stop Condition
                    if global_failure_rate <= target_failure_rate:
                        status.update(label="Optimization Successful!", state="complete", expanded=False)
                        st.toast(f"Converged in {iteration} iterations!")
                        break

                        # 7. Feedback Loop - Intelligence Layer
                        if iteration < Config.MAX_OPTIMIZATION_ITERATIONS:
                            # Identify Failed Routes
                            failed_route_ids = set()

                            for _, day_row in sim_results.iterrows():
                                routes = day_row.get('Route_Breakdown', [])
                                # Check if any route on this day failed
                                for route_data in routes:
                                    if not route_data.get('success', True):
                                        failed_route_ids.add(route_data['route_id'])

                            # Target Risky Customers - Smart Group Punishment
                            penalized_customers = 0

                            for route_idx, route in enumerate(solution['routes']):
                                if route_idx in failed_route_ids:
                                    # Get customers on this failed route (excluding depot)
                                    route_customers = [node for node in route[1:-1] if node > 0]  # Skip depot (0) and any invalid nodes

                                    # Apply penalties to volatile customers on this route
                                    route_penalized = False

                                    for customer_idx in route_customers:
                                        # Convert matrix index back to customer data index
                                        # Matrix index = data index + 1 (because depot is 0)
                                        if customer_idx > 0:
                                            data_idx = customer_idx - 1

                                            if data_idx < len(st.session_state.data):
                                                customer_row = st.session_state.data.iloc[data_idx]
                                                customer_id = customer_row.get('מס\' לקוח')

                                                if customer_id is not None:
                                                    # Check volatility (std_volume)
                                                    std_volume = customer_row.get('std_volume', 0)

                                                    if std_volume > 0.1:  # Ignore noise, focus on truly volatile customers
                                                        # Apply penalty: increase multiplier
                                                        current_mult = current_multipliers.get(customer_id, 1.0)
                                                        new_mult = current_mult + 0.5
                                                        current_multipliers[customer_id] = min(new_mult, 3.0)  # Clamp at 3.0 max
                                                        penalized_customers += 1
                                                        route_penalized = True

                                    # Fallback: if no volatile customers on route, apply smaller penalty to all
                                    if not route_penalized and route_customers:
                                        for customer_idx in route_customers:
                                            data_idx = customer_idx - 1
                                            if data_idx < len(st.session_state.data):
                                                customer_row = st.session_state.data.iloc[data_idx]
                                                customer_id = customer_row.get('מס\' לקוח')

                                                if customer_id is not None:
                                                    current_mult = current_multipliers.get(customer_id, 1.0)
                                                    new_mult = current_mult + 0.2  # Smaller penalty
                                                    current_multipliers[customer_id] = min(new_mult, 3.0)
                                                    penalized_customers += 1

                            # Report intelligence actions
                            if penalized_customers > 0:
                                status.write(f"🧠 Intelligence: Penalized {penalized_customers} customers on {len(failed_route_ids)} failed routes.")
                            else:
                                status.write("🧠 Intelligence: No specific customers penalized (continuing with current plan).")
                        else:
                            status.update(label="Max iterations reached - Target not met.", state="error", expanded=False)
                            st.warning(f"Optimization finished but failure rate ({global_failure_rate:.1f}%) exceeds target ({target_failure_rate}%).")
                    else:
                        st.error(f"Solver failed to find a feasible solution in Iteration {iteration}. Try increasing fleet size or Time Limits.")
                        break  # Stop the loop immediately

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

            # Expandable Route Details
            st.subheader("Route Details")

            # Retrieve volume limits from fleet settings
            small_truck_vol = services['fleet_settings']['small_truck_vol']
            big_truck_vol = services['fleet_settings']['big_truck_vol']

            route_metrics = solution.get('route_metrics', [])

            # Routes per Day view (default)
            # Check if simulation results exist and are not empty
            if st.session_state.get('simulation_results') is not None and not st.session_state.simulation_results.empty:
                sim_results = st.session_state.simulation_results
                
                # Retrieve Safety Factor
                safety_factor = services['fleet_settings']['safety_factor']

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
                                'Time (h)': route_data['duration_seconds'] / 3600,     # Convert to hours
                                'success': route_data.get('success', True),  # Include success status
                                'status': route_data.get('status', 'OK')     # Include status string
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

                    # Calculate success rate statistics
                    total_days_driven = len(route_daily_data)
                    success_days = sum(1 for day in route_daily_data if day.get('success', True))
                    success_rate = (success_days / total_days_driven * 100) if total_days_driven > 0 else 0.0

                    # Calculate averages for this route
                    avg_stops = np.mean([day['Stops'] for day in route_daily_data])
                    avg_volume = np.mean([day['Volume'] for day in route_daily_data]) / safety_factor
                    avg_distance = np.mean([day['Distance (km)'] for day in route_daily_data])
                    avg_time = np.mean([day['Time (h)'] for day in route_daily_data])

                    # Count days this route was driven
                    days_driven = len(route_daily_data)

                    # Choose emoji based on success rate
                    success_emoji = "✅" if success_rate >= 95.0 else "⚠️"

                    # Expander header with success rate, averages and days driven
                    header = f"Route {idx+1}: {success_emoji} {success_rate:.1f}% Success | Avg {avg_stops:.1f} stops | Avg Vol: {avg_volume:.1f} | Avg Dist: {avg_distance:.1f} km | Avg Time: {avg_time:.1f} h | Driven: {days_driven}/{total_sim_days} days"

                    with st.expander(header):
                        # Create DataFrame for daily stats
                        daily_df = pd.DataFrame(route_daily_data)
                        # Format Date column to clean string format
                        if 'Date' in daily_df.columns:
                            daily_df['Date'] = pd.to_datetime(daily_df['Date']).dt.strftime('%Y-%m-%d')
                        # Ensure status column is included and formatted
                        if 'status' in daily_df.columns:
                            # Reorder columns to put status at the end for better visibility
                            cols = [col for col in daily_df.columns if col != 'status'] + ['status']
                            daily_df = daily_df[cols]
                        st.dataframe(daily_df, use_container_width=True, hide_index=True)

                # Footer: System-wide daily averages and totals
                if not sim_results.empty:
                    system_avg_distance = sim_results['Total_Distance_km'].mean()
                    system_avg_time = sim_results['Daily_Total_Duration'].mean() / 3600.0  # Convert seconds to hours
                    avg_stops_daily = sim_results['Total_Stops'].mean()
                    
                    # Calculate total volume usage
                    total_load_sum = sim_results['Daily_Total_Load'].sum()
                    total_capacity_sum = sim_results['Daily_Total_Capacity'].sum()
                    total_volume_usage = (total_load_sum / total_capacity_sum * 100) if total_capacity_sum > 0 else 0.0

                    # Calculate grand totals for the entire simulation period
                    total_distance_all_days = sim_results['Total_Distance_km'].sum()
                    total_time_all_days = sim_results['Max_Shift_Duration_hours'].sum()

                    st.markdown("---")
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    col1.metric("System Avg Daily Distance", f"{system_avg_distance:.1f} km")
                    col2.metric("Total Daily Distance", f"{total_distance_all_days:.1f} km")
                    col3.metric("Avg Daily Man-Hours", f"{system_avg_time:.1f} h")
                    col4.metric("Total Daily Time", f"{total_time_all_days:.1f} hours")
                    col5.metric("Avg Daily Stops", f"{avg_stops_daily:.1f}")
                    col6.metric("Total Volume Usage", f"{total_volume_usage:.1f}%")
            else:
                # Edge case: simulation results not available
                st.info("Daily simulation statistics are unavailable. Please check the 'Static Routes' tab for route details.")

        else:
            st.error("Optimization failed.")
