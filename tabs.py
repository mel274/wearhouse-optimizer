"""
Tab handlers for the Warehouse Optimizer Streamlit app.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from data_manager import DataManager
from calculations.route_optimizer import RouteOptimizer
from calculations.logistics import calculate_fleet_metrics
from visualizer import MapBuilder
import utils


def tab_data_upload(services: Optional[Dict[str, Any]]) -> None:
    """Handle the data upload tab functionality."""
    st.header("Data Upload")
    
    # Handle case where services might be None (shouldn't happen after fix, but defensive)
    if services is None:
        st.warning("Application configuration error. Please refresh the page.")
        return

    # Initialize services if not already done
    if services['data_manager'] is None:
        services['data_manager'] = DataManager()
    if services['map_builder'] is None:
        services['map_builder'] = MapBuilder()

    uploaded_file = st.file_uploader("Select Excel/CSV", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        try:
            file_id = f"{uploaded_file.name}-{uploaded_file.size}"
            # Reload data if file changed OR if raw_data is missing
            if (st.session_state.file_id != file_id or 
                not hasattr(st.session_state, 'raw_data') or 
                st.session_state.raw_data is None):
                raw_data = services['data_manager'].load_data(uploaded_file)
                # Preserve raw data before aggregation (needed for date filtering in comparison)
                st.session_state.raw_data = raw_data.copy()
                aggregated_data = services['data_manager'].aggregate_data(raw_data)
                st.session_state.data = aggregated_data
                st.session_state.file_id = file_id
                st.session_state.solution = None
                st.session_state.warehouse_coords = None
                # Clear simulation data when new file is uploaded
                st.session_state.distance_matrix = None
                st.session_state.time_matrix = None
                st.session_state.node_map = None

            if st.session_state.data is not None:
                # Hide technical columns from user view
                display_columns = [col for col in st.session_state.data.columns 
                                 if col not in ['color', 'truck_type', 'trucks_needed']]
                st.dataframe(st.session_state.data[display_columns])
                geocoded_count = st.session_state.data[['lat', 'lng']].notna().all(axis=1).sum()

                # Geocode button - disabled if geo_service is not available
                geocode_disabled = st.session_state.geo_service is None
                
                if st.button("Geocode Addresses", type="primary", disabled=geocode_disabled):
                    if st.session_state.geo_service:
                        with st.spinner("Geocoding..."):
                            geocoded_data = services['data_manager'].add_coordinates(
                                st.session_state.data, st.session_state.geo_service
                            )
                            st.session_state.data = geocoded_data
                            
                            # Calculate logistics after geocoding
                            st.session_state.data = calculate_fleet_metrics(
                                st.session_state.data,
                                services['fleet_settings']
                            )
                            
                            st.rerun()
                    else:
                        st.error("Geocoding service is not available. Please configure an API key.")

                # Show map immediately after geocoding (if coordinates are available)
                if geocoded_count > 0:
                    st.subheader("Customer Map")
                    
                    # Check for manual warehouse address
                    warehouse_cog = None
                    manual_warehouse_address = services.get('warehouse_address')
                    
                    if manual_warehouse_address and st.session_state.geo_service:
                        # Geocode the manual warehouse address
                        with st.spinner("Geocoding warehouse address..."):
                            wh_coords = st.session_state.geo_service.get_coordinates(manual_warehouse_address)
                            if wh_coords:
                                warehouse_cog = {
                                    'lat': wh_coords[0],
                                    'lng': wh_coords[1]
                                }
                                # Store in session state for use in optimization
                                st.session_state.warehouse_coords = wh_coords
                    
                    # Pass None if no manual warehouse address (map will show only customers)
                    phase1_map = services['map_builder'].create_phase1_map(st.session_state.data, warehouse_cog)
                    st.components.v1.html(phase1_map._repr_html_(), height=600)
                
                # Check for failed geocoding (rows with missing lat/lng) - show regardless of map display
                failed_geocoding = st.session_state.data[
                    st.session_state.data[['lat', 'lng']].isna().any(axis=1)
                ].copy()
                
                if len(failed_geocoding) > 0:
                    st.warning(f"Customers with Invalid Addresses ({len(failed_geocoding)})")
                    
                    # Display only relevant columns for identification
                    display_cols = []
                    if "מס' לקוח" in failed_geocoding.columns:
                        display_cols.append("מס' לקוח")
                    if "שם לקוח" in failed_geocoding.columns:
                        display_cols.append("שם לקוח")
                    if "כתובת" in failed_geocoding.columns:
                        display_cols.append("כתובת")
                    
                    if display_cols:
                        st.dataframe(failed_geocoding[display_cols], width="stretch")
                    else:
                        st.dataframe(failed_geocoding, width="stretch")

        except Exception as e:
            st.error(f"Error: {str(e)}")


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
    from config import Config
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

    # Auto-calculate capacity based on volume
    total_volume = st.session_state.data['total_volume_m3'].sum()
    total_quantity = st.session_state.data['total_quantity'].sum()

    if total_quantity > 0:
        avg_tire_vol = total_volume / total_quantity
    else:
        avg_tire_vol = 0.00001  # Fallback to prevent division by zero

    big_truck_vol = services['fleet_settings']['big_truck_vol']
    small_truck_vol = services['fleet_settings']['small_truck_vol']
    safety_factor = services['fleet_settings']['safety_factor']
    
    # Calculate separate capacities for big and small trucks
    big_capacity = int((big_truck_vol * safety_factor) / avg_tire_vol) if avg_tire_vol > 0.00001 else 0
    small_capacity = int((small_truck_vol * safety_factor) / avg_tire_vol) if avg_tire_vol > 0.00001 else 0
    
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
                from calculations.simulation import create_node_map
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

        except Exception as e:
            st.error(f"Optimization Error: {str(e)}")
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Optimization error: {e}")


def tab_export() -> None:
    """Handle the export results tab functionality."""
    st.header("Export")
    if st.session_state.solution and st.session_state.solution['solution_found']:
        try:
            solution = st.session_state.solution
            
            if st.session_state.data is None:
                st.warning("No customer data available for export.")
                return

            # Build clean route list
            routes_data = []
            
            # Export only served customers (routes)
            for route_idx, route in enumerate(solution['routes']):
                # Skip empty routes (only depot)
                if len(route) <= 2:
                    continue
                
                # Process stops in route (skip depot at start and end)
                for stop_seq, node_idx in enumerate(route[1:-1], start=1):
                    if node_idx > 0 and node_idx - 1 < len(st.session_state.data):
                        customer = st.session_state.data.iloc[node_idx - 1]
                        routes_data.append({
                            'Route ID': route_idx + 1,
                            'Stop Sequence': stop_seq,
                            'Customer ID': str(customer.get("מס' לקוח", "")),
                            'Customer Name': customer.get('שם לקוח', ''),
                            'Address': customer.get('כתובת', ''),
                            'Quantity': int(customer.get('total_quantity', 0))
                        })

            if not routes_data:
                st.warning("No routes to export.")
                return

            # Create DataFrame
            routes_df = pd.DataFrame(routes_data)
            
            # Display preview
            st.dataframe(routes_df, width="stretch")

            # Export to Excel
            from io import BytesIO
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                routes_df.to_excel(writer, index=False, sheet_name='Routes')
            
            output.seek(0)
            
            # Download button
            st.download_button(
                "Download Routes (Excel)",
                output.getvalue(),
                "optimization_routes.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='download-excel'
            )

        except Exception as e:
            st.error(f"Export Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning("No solution to export.")


def tab_compare_actuals(services: Optional[Dict[str, Any]]) -> None:
    """Handle the performance comparison tab functionality."""
    st.header("Comparison")
    
    # Handle case where services might be None (shouldn't happen after fix, but defensive)
    if services is None:
        st.warning("Application configuration error. Please refresh the page.")
        return

    if not st.session_state.solution or not st.session_state.solution.get('solution_found'):
        st.warning("Please run an optimization first before comparing results.")
        return

    # Check if simulation data is available
    if not hasattr(st.session_state, 'distance_matrix') or not hasattr(st.session_state, 'time_matrix'):
        st.warning("Simulation data not available. Please run optimization again to enable historical backtesting.")
        return

    # Check if main order data is available
    if st.session_state.data is None:
        st.warning("Please upload order data in the Data Upload tab first.")
        return

    # Check if raw data is available (needed for date filtering)
    if not hasattr(st.session_state, 'raw_data') or st.session_state.raw_data is None:
        st.warning("Raw order data is not available. Please re-upload your order file in the Data Upload tab to enable date filtering.")
        return

    # Summary File Upload (for projected actuals calculation)
    st.subheader("Summary File Upload")
    
    summary_file = st.file_uploader("Upload Summary File (CSV/Excel)", type=['csv', 'xlsx', 'xls'], key='summary_file')

    # Date Range Picker
    st.subheader("Date Range Selection")
    
    # Get available date range from raw data
    order_date_col = 'תאריך אספקה'
    if order_date_col not in st.session_state.raw_data.columns:
        st.error(f"Order data must contain a date column: '{order_date_col}'")
        return
    
    raw_data = st.session_state.raw_data.copy()
    raw_data[order_date_col] = pd.to_datetime(raw_data[order_date_col], dayfirst=True, errors='coerce')
    raw_data = raw_data[raw_data[order_date_col].notna()]
    
    if len(raw_data) == 0:
        st.error("No valid dates found in order data.")
        return
    
    min_date = raw_data[order_date_col].min().date()
    max_date = raw_data[order_date_col].max().date()
    
    # Single date range picker
    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Validate date range selection
    # st.date_input with tuple value returns a tuple when both dates are selected
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.error("Please select both start and end dates.")
        return
    
    start_date, end_date = date_range[0], date_range[1]
    
    if start_date > end_date:
        st.error("Start date must be before end date.")
        return

    # Filter raw data by selected date range
    start_date_dt = pd.Timestamp(start_date)
    end_date_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    filtered_data = raw_data[
        (raw_data[order_date_col] >= start_date_dt) & 
        (raw_data[order_date_col] <= end_date_dt)
    ].copy()

    if len(filtered_data) == 0:
        st.warning(f"No orders found in the selected date range.")
        return

    # Load summary file if provided
    df_summary = None
    summary_total_km = None
    summary_total_hours = None
    total_days_in_summary = None
    
    if summary_file:
        try:
            df_summary = pd.read_csv(summary_file) if summary_file.name.endswith('.csv') else pd.read_excel(summary_file)
            
            # Check for summary metrics columns
            if 'סהכ קמ' in df_summary.columns:
                summary_total_km = df_summary['סהכ קמ'].sum()
            elif 'Total KM' in df_summary.columns:
                summary_total_km = df_summary['Total KM'].sum()
            
            if 'סהכ זמן נסיעה' in df_summary.columns:
                summary_total_hours = df_summary['סהכ זמן נסיעה'].sum()
            elif 'Total Driving Time' in df_summary.columns:
                summary_total_hours = df_summary['Total Driving Time'].sum()
            
            # Calculate total days in summary (use date range from summary file if available)
            date_range_col = 'טווח תאריכים'
            if date_range_col in df_summary.columns:
                date_range_str = df_summary[date_range_col].iloc[0]
                if pd.notna(date_range_str):
                    try:
                        date_range_str = str(date_range_str).strip()
                        if date_range_str.startswith('[') and date_range_str.endswith(']'):
                            date_range_str = date_range_str[1:-1].strip()
                        date_parts = date_range_str.split('-')
                        if len(date_parts) == 2:
                            start_str = date_parts[0].strip().replace('.', '/')
                            end_str = date_parts[1].strip().replace('.', '/')
                            summary_start = pd.to_datetime(start_str, dayfirst=True, errors='coerce')
                            summary_end = pd.to_datetime(end_str, dayfirst=True, errors='coerce')
                            if pd.notna(summary_start) and pd.notna(summary_end):
                                total_days_in_summary = (summary_end - summary_start).days + 1
                    except:
                        pass
            
            if total_days_in_summary is None:
                # Fallback: use the selected date range
                total_days_in_summary = (end_date - start_date).days + 1
        except Exception as e:
            st.warning(f"Could not load summary file: {str(e)}")

    # Merge coordinates from aggregated data back into filtered raw data
    aggregated_data = st.session_state.data.copy()
    customer_id_col = "מס' לקוח"
    
    if customer_id_col in aggregated_data.columns and customer_id_col in filtered_data.columns:
        # Create a mapping of customer ID to coordinates
        coord_map = {}
        for idx, row in aggregated_data.iterrows():
            customer_id = row.get(customer_id_col)
            if pd.notna(customer_id) and pd.notna(row.get('lat')) and pd.notna(row.get('lng')):
                coord_map[customer_id] = {
                    'lat': row['lat'],
                    'lng': row['lng']
                }
        
        # Merge coordinates into filtered data
        filtered_data['lat'] = filtered_data[customer_id_col].map(lambda x: coord_map.get(x, {}).get('lat') if pd.notna(x) else None)
        filtered_data['lng'] = filtered_data[customer_id_col].map(lambda x: coord_map.get(x, {}).get('lng') if pd.notna(x) else None)
    else:
        st.error("Could not merge coordinates. Customer ID column may be missing.")
        return

    # Filter out rows without coordinates (needed for simulation)
    filtered_data_with_coords = filtered_data[
        filtered_data[['lat', 'lng']].notna().all(axis=1)
    ].copy()

    if len(filtered_data_with_coords) == 0:
        st.error("No orders with valid coordinates found. Please ensure your data has been geocoded.")
        return

    # Run simulation on filtered data with coordinates
    from calculations.simulation import run_historical_simulation
    
    with st.spinner("Running simulation..."):
        simulation_results = run_historical_simulation(
            historical_data=filtered_data_with_coords,
            master_routes=st.session_state.solution['routes'],
            distance_matrix=st.session_state.distance_matrix,
            time_matrix=st.session_state.time_matrix,
            node_map=st.session_state.node_map,
            service_time_seconds=st.session_state.optimization_params['service_time_seconds'],
            date_col=order_date_col,
            customer_id_col=customer_id_col,
            depot=0
        )
    
    # Convert Date column to datetime for filtering
    simulation_results['Date'] = pd.to_datetime(simulation_results['Date'])
    
    # Filter simulation results by selected date range
    filtered_simulation = simulation_results[
        (simulation_results['Date'].dt.date >= start_date) &
        (simulation_results['Date'].dt.date <= end_date)
    ].copy()

    num_selected_days = len(filtered_simulation)
    
    if num_selected_days == 0:
        st.warning("No simulation results found for the selected date range.")
        return

    # Calculate daily averages from summary file
    avg_actual_km_per_day = None
    avg_actual_time_per_day = None
    if summary_total_km is not None and total_days_in_summary is not None and total_days_in_summary > 0:
        avg_actual_km_per_day = summary_total_km / total_days_in_summary
    if summary_total_hours is not None and total_days_in_summary is not None and total_days_in_summary > 0:
        avg_actual_time_per_day = summary_total_hours / total_days_in_summary

    # Calculate projected and simulated totals for selected days
    projected_actual_km = None
    projected_actual_time = None
    if avg_actual_km_per_day is not None:
        projected_actual_km = avg_actual_km_per_day * num_selected_days
    if avg_actual_time_per_day is not None:
        projected_actual_time = avg_actual_time_per_day * num_selected_days

    simulated_km = filtered_simulation['Total_Distance_km'].sum()
    simulated_time = filtered_simulation['Max_Shift_Duration_hours'].sum()  # Total time in hours

    # Fleet Utilization Comparison
    st.subheader("Fleet Utilization")
    
    # System metric: Average Active_Routes from simulation
    system_avg_trucks = filtered_simulation['Active_Routes'].mean()
    
    # Historical metric: Count unique route names per day from filtered raw data
    historical_avg_trucks = None
    route_name_col = 'שם קו'
    
    if route_name_col in filtered_data.columns:
        filtered_data['Date'] = pd.to_datetime(filtered_data[order_date_col], dayfirst=True, errors='coerce')
        historical_for_selected = filtered_data[
            (filtered_data['Date'].dt.date >= start_date) &
            (filtered_data['Date'].dt.date <= end_date)
        ].copy()
        
        if len(historical_for_selected) > 0:
            # Group by date and count unique route names
            daily_route_counts = historical_for_selected.groupby(
                historical_for_selected['Date'].dt.date
            )[route_name_col].nunique()
            
            if len(daily_route_counts) > 0:
                historical_avg_trucks = daily_route_counts.mean()
    
    # Display fleet utilization metrics side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.metric("System Avg Trucks", f"{system_avg_trucks:.1f}")
    with col2:
        if historical_avg_trucks is not None:
            st.metric("Historical Avg Trucks", f"{historical_avg_trucks:.1f}")
        else:
            st.metric("Historical Avg Trucks", "N/A")

    # Comparison Table
    st.subheader("Comparison: Projected vs. Optimized")
    
    comparison_data = []
    
    # Row 1: Distance
    if projected_actual_km is not None:
        km_diff = simulated_km - projected_actual_km
        comparison_data.append({
            'Metric': 'Distance (km)',
            'Projected Actual': f"{projected_actual_km:,.1f}",
            'Simulated (Optimized)': f"{simulated_km:,.1f}",
            'Difference': f"{km_diff:+,.1f}"
        })
    
    # Row 2: Time
    if projected_actual_time is not None:
        time_diff = simulated_time - projected_actual_time
        comparison_data.append({
            'Metric': 'Time (hours)',
            'Projected Actual': f"{projected_actual_time:,.1f}",
            'Simulated (Optimized)': f"{simulated_time:,.1f}",
            'Difference': f"{time_diff:+,.1f}"
        })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df.set_index('Metric'), width="stretch")
