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
from calculations.simulation import create_node_map, run_historical_simulation, count_simulation_exceptions
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
    
    # Get exception budget parameter
    available_exception_percent = services.get('available_exception_percent', Config.AVAILABLE_EXCEPTION_PERCENT)

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
            # Prepare common data for optimization
            matrix_coords = [wh_coords] + list(zip(valid_coords['lat'], valid_coords['lng']))
            demands = [0] + estimated_demand.tolist()
            base_shift_seconds = int(round(services['max_shift_hours'] * 3600))

            # Exception Budget Retry Loop
            # Fleet Squeeze finds minimum trucks, then simulation checks budget
            # If budget exceeded, retry forcing more trucks until budget met or fleet exhausted
            total_fleet_size = num_small_trucks + num_big_trucks
            min_trucks_required = None  # Let Fleet Squeeze calculate on first attempt
            final_solution = None
            final_sim_results = None
            budget_met = False

            progress_placeholder = st.empty()

            params = {
                'fleet_size': total_fleet_size,
                'num_small_trucks': num_small_trucks,
                'num_big_trucks': num_big_trucks,
                'small_capacity': small_capacity,
                'big_capacity': big_capacity,
                'max_shift_seconds': base_shift_seconds,
                'service_time_seconds': services['service_time_minutes'] * 60
            }

            while True:
                if min_trucks_required is not None:
                    progress_placeholder.info(f"Retrying with minimum {min_trucks_required} trucks...")
                    logger.info(f"Optimization attempt: forcing minimum {min_trucks_required} trucks")
                else:
                    progress_placeholder.info(f"Optimizing with Fleet Squeeze...")
                    logger.info(f"Optimization attempt: Fleet Squeeze with {total_fleet_size} available trucks")

                # CALL OPTIMIZER
                solution = services['route_optimizer'].solve(
                    [f"Loc {i}" for i in range(len(matrix_coords))],
                    demands,
                    params,
                    coords=matrix_coords,
                    customer_details=customer_details,
                    min_trucks_override=min_trucks_required
                )

                if not solution['solution_found']:
                    logger.warning(f"Optimization failed")
                    break

                # Store matrices and node map
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

                # Run simulation to check exception budget
                required_session_keys = ['raw_data', 'distance_matrix', 'time_matrix', 'node_map']
                missing_keys = [key for key in required_session_keys if not hasattr(st.session_state, key) or getattr(st.session_state, key) is None]

                if missing_keys:
                    logger.warning(f"Cannot run simulation, missing: {missing_keys}")
                    final_solution = solution
                    budget_met = True
                    break

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

                # Check exception budget
                exception_count, total_drives = count_simulation_exceptions(sim_results)
                exception_budget = available_exception_percent * total_drives

                logger.info(f"Exception check: {exception_count} exceptions vs budget {exception_budget:.1f} ({available_exception_percent*100:.1f}% of {total_drives} drives)")

                if exception_count <= exception_budget:
                    # Budget met - success!
                    final_solution = solution
                    final_sim_results = sim_results
                    budget_met = True
                    trucks_used = solution['metrics']['num_vehicles_used']
                    progress_placeholder.success(f"Success with {trucks_used} trucks! {exception_count} exceptions within budget of {exception_budget:.1f} ({available_exception_percent*100:.1f}%)")
                    logger.info(f"Exception budget met with {trucks_used} trucks")
                    break
                else:
                    # Budget exceeded - try with more trucks
                    trucks_used = solution['metrics']['num_vehicles_used']
                    progress_placeholder.warning(f"{trucks_used} trucks: {exception_count} exceptions exceed budget {exception_budget:.1f}. Retrying...")
                    logger.info(f"Exception budget exceeded: {exception_count} > {exception_budget:.1f}. Retrying with more trucks.")
                    final_solution = solution  # Keep last solution in case we hit fleet max
                    final_sim_results = sim_results
                    
                    # Set minimum trucks for next attempt
                    if min_trucks_required is None:
                        min_trucks_required = trucks_used + 1
                    else:
                        min_trucks_required += 1
                    
                    # Check if we've exhausted the fleet
                    if min_trucks_required > total_fleet_size:
                        logger.warning(f"Fleet exhausted: cannot use more than {total_fleet_size} trucks")
                        break

            # Store final results
            if final_solution:
                st.session_state.solution = final_solution
                st.session_state.optimization_params = params
                st.session_state.valid_coords_for_simulation = valid_coords

                if final_sim_results is not None:
                    st.session_state.simulation_results = final_sim_results

                if not budget_met:
                    st.warning(f"Fleet limit reached ({total_fleet_size} trucks). Solution found but exception budget may be exceeded. Consider adjusting constraints.")
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

            # Unserved Customers Warning
            if unserved_list:
                unserved_ids = [str(u.get('id', 'Unknown')) for u in unserved_list]
                st.error(f"⚠️ Unserved Customers Detected: {', '.join(unserved_ids)}")

            # MAP Visualization
            st.subheader("Route Map")
            phase2_map = services['map_builder'].create_phase2_map(
                solution, valid_coords, wh_coords,
                geo_service=st.session_state.geo_service
            )
            st.components.v1.html(phase2_map._repr_html_(), height=700)

            # Unified Failed Customers Display
            removed_customers = solution.get('removed_customers', [])
            skipped_indices = solution.get('skipped_customers', [])
            
            if removed_customers or skipped_indices:
                total_failed = len(removed_customers) + len(skipped_indices)
                with st.expander(f"Unroutable Customers Details ({total_failed} customers)", expanded=False):
                    failed_data = []
                    
                    # Add removed customers (optimization phase)
                    for removed in removed_customers:
                        failed_data.append({
                            'Customer Name': removed.get('name', 'Unknown'),
                            'Address': removed.get('address', 'Unknown'),
                            'Error Reason': 'Removed during optimization (not included in routes)'
                        })
                    
                    # Add skipped customers (visualization phase)
                    for node_idx in skipped_indices:
                        customer_row_idx = node_idx - 1  # Node 0 is warehouse
                        if 0 <= customer_row_idx < len(valid_coords):
                            row = valid_coords.iloc[customer_row_idx]
                            failed_data.append({
                                'Customer Name': row.get('שם לקוח', f'Customer {node_idx}'),
                                'Address': row.get('כתובת', row.get('Address', 'Unknown')),
                                'Error Reason': 'Skipped during visualization (included in routes but unroutable)'
                            })
                        else:
                            failed_data.append({
                                'Customer Name': f'Customer {node_idx}',
                                'Address': 'Unknown',
                                'Error Reason': 'Skipped during visualization (included in routes but unroutable)'
                            })
                    
                    if failed_data:
                        failed_df = pd.DataFrame(failed_data)
                        st.dataframe(failed_df, width='stretch', hide_index=True)

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
