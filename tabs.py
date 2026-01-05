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
    st.header("📊 Data Upload")
    
    # Handle case where services might be None (shouldn't happen after fix, but defensive)
    if services is None:
        st.warning("⚠️ Application configuration error. Please refresh the page.")
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
                st.success(f"Loaded {len(aggregated_data)} customers")

            if st.session_state.data is not None:
                st.dataframe(st.session_state.data)
                geocoded_count = st.session_state.data[['lat', 'lng']].notna().all(axis=1).sum()

                # Geocode button - disabled if geo_service is not available
                geocode_disabled = st.session_state.geo_service is None
                if geocode_disabled:
                    st.info("ℹ️ Geocoding requires an API key. Please configure it in your environment variables.")
                
                if st.button("📍 Geocode Addresses", type="primary", disabled=geocode_disabled):
                    if st.session_state.geo_service:
                        with st.spinner("Geocoding..."):
                            geocoded_data = services['data_manager'].add_coordinates(
                                st.session_state.data, st.session_state.geo_service
                            )
                            st.session_state.data = geocoded_data
                            st.rerun()
                    else:
                        st.error("Geocoding service is not available. Please configure an API key.")

                if geocoded_count > 0:
                    # Calculate logistics
                    with st.spinner("Calculating fleet logistics..."):
                        st.session_state.data = calculate_fleet_metrics(
                            st.session_state.data,
                            services['fleet_settings']
                        )

                    st.subheader("📈 Center of Gravity & Logistics Analysis")

                    # Display Metrics
                    total_volume = st.session_state.data['total_volume_m3'].sum()
                    total_trucks = st.session_state.data['trucks_needed'].sum()

                    col1, col2 = st.columns(2)
                    col1.metric("Total Volume (m³)", f"{total_volume:.2f}")
                    col2.metric("Estimated Trucks Needed", f"{total_trucks}")

                    # Optional COG Calculation
                    calculate_cog = st.toggle(
                        "Auto-Calculate Warehouse Location (COG)",
                        value=False,
                        help="If enabled, calculates the geographic center of all customers to suggest a warehouse location."
                    )

                    if calculate_cog:
                        with st.spinner("Calculating Center of Gravity..."):
                            cog = services['data_manager'].calculate_center_of_gravity(st.session_state.data)

                            if cog['lat'] and st.session_state.geo_service:
                                raw_coords = (cog['lat'], cog['lng'])
                                snapped_coords = st.session_state.geo_service.snap_to_road(raw_coords)

                                if snapped_coords != raw_coords:
                                    cog['lat'], cog['lng'] = snapped_coords
                                    st.info(f"ℹ️ Warehouse Auto-Snapped to nearest road: {snapped_coords}")

                            st.session_state.warehouse_coords = (cog['lat'], cog['lng'])

                            # Show the warehouse and customer locations on the map
                            if cog['lat'] is not None:
                                st.subheader("🗺️ Customer & Warehouse Map")
                                phase1_map = services['map_builder'].create_phase1_map(st.session_state.data, cog)
                                st.components.v1.html(phase1_map._repr_html_(), height=600)
                    else:
                        # Ensure warehouse_coords are reset if COG is not calculated
                        st.session_state.warehouse_coords = None

        except Exception as e:
            st.error(f"Error: {str(e)}")


def tab_optimization(services: Optional[Dict[str, Any]]) -> None:
    """Handle the route optimization tab functionality."""
    st.header("🚛 Route Optimization")
    
    # Handle case where services might be None (shouldn't happen after fix, but defensive)
    if services is None:
        st.warning("⚠️ Application configuration error. Please refresh the page.")
        return
    
    if st.session_state.data is None:
        st.warning("Please upload data first in the Data Upload tab.")
        return

    # Check if API key is available for optimization
    from config import Config
    if not Config.OPENROUTESERVICE_API_KEY:
        st.warning("⚠️ **Optimization requires an API Key.** Please configure `OPENROUTESERVICE_API_KEY` in your environment variables to use this feature.")
        st.info("You can still upload and view data in the Data Upload tab without an API key.")
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
    safety_factor = services['fleet_settings']['safety_factor']
    calculated_capacity = int((big_truck_vol * safety_factor) / avg_tire_vol) if avg_tire_vol > 0.00001 else 0

    st.info(f"""ℹ️ **Auto-Calculated Capacity:** `{calculated_capacity}` tires/truck
**Fleet Status:** 18 Trucks Available (12 Big, 6 Small)
**Calculation Basis:** Big Truck ({big_truck_vol}m³) & Avg Tire Volume ({avg_tire_vol:.4f}m³)""")

    if st.button("🚀 Run Auto-Optimization", type="primary"):
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
            # Fallback to COG-calculated coordinates
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
                    'fleet_size': services['fleet_size'],
                    'capacity': calculated_capacity,
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
                        st.warning(f"Optimization Complete with Warnings: {num_routes} trucks used. {len(unserved_list)} customers could not be served.")

                        # Display suggestions generated by the optimizer
                        if 'suggestions' in solution and not solution['suggestions'].get('all_served', True):
                            st.subheader("💡 Optimization Suggestions")
                            suggestions = solution['suggestions']

                            # Prioritize fleet utilization suggestion
                            if num_routes < params['fleet_size']:
                                st.info(f"**Suggestion**: Only {num_routes} of {params['fleet_size']} available trucks were used. The remaining unserved customers could not be formed into a valid route within the current constraints (e.g., shift time). Consider extending the shift time.")

                            # Display detailed suggestions from the optimizer
                            for suggestion in suggestions.get('suggestions', []):
                                with st.expander(f"🔹 {suggestion['message']}", expanded=False):
                                    st.write(suggestion.get('suggestion', ''))
                                    if 'details' in suggestion:
                                        st.json(suggestion['details'], expanded=False)

                        # Show Unserved Customers
                        with st.expander(f"⚠️ Unserved Customers ({len(unserved_list)})", expanded=False):
                            unserved_data = []
                            for u in unserved_list:
                                if u['id'] - 1 < len(valid_coords):
                                    cust = valid_coords.iloc[u['id'] - 1]
                                    unserved_data.append({
                                        "ID": cust.get("מס' לקוח", ""),
                                        "Name": cust.get("שם לקוח", ""),
                                        "Demand": u['demand'],
                                        "Reason": u['reason']
                                    })
                            if unserved_data:
                                st.dataframe(pd.DataFrame(unserved_data))
                    else:
                        st.success(f"Optimization Complete: {num_routes} trucks used. All customers served.")

                    # MAP Visualization
                    st.subheader("🗺️ Route Map")
                    phase2_map = services['map_builder'].create_phase2_map(
                        solution, valid_coords, wh_coords,
                        geo_service=st.session_state.geo_service
                    )
                    st.components.v1.html(phase2_map._repr_html_(), height=700)

                    # Expandable Route Details
                    st.subheader("📋 Route Details")

                    route_metrics = solution.get('route_metrics', [])

                    for idx, route in enumerate(solution['routes']):
                        if len(route) <= 2: continue

                        current_load = sum(demands[node] for node in route)
                        r_metrics = route_metrics[idx] if idx < len(route_metrics) else {'distance': 0, 'duration': 0}
                        dist_km = r_metrics.get('distance', 0) / 1000
                        dur_str = utils.format_time(r_metrics.get('duration', 0))

                        warnings = []
                        if current_load > calculated_capacity: warnings.append("⚠️ Overloaded")
                        if r_metrics.get('duration', 0) > params['max_shift_seconds']: warnings.append("⚠️ Overtime")
                        status_str = " ".join(warnings) if warnings else "✅ OK"

                        header = f"Truck {idx+1}: {len(route)-2} Stops | Load: {current_load} | Dist: {dist_km:.1f} km | Time: {dur_str} {status_str}"

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
    st.header("📊 Optimization Results")
    if st.session_state.solution and st.session_state.solution['solution_found']:
        try:
            solution = st.session_state.solution
            metrics = solution.get('metrics', {})

            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Customers Served", f"{metrics.get('customers_served', 0)}")
            with col2:
                st.metric("Vehicles Used", f"{metrics.get('num_vehicles_used', 0)}")
            with col3:
                st.metric("Total Distance", f"{metrics.get('total_distance', 0)/1000:.1f} km")

            # Display optimization suggestions if any
            if 'suggestions' in solution and not solution['suggestions'].get('all_served', True):
                st.subheader("🔍 Optimization Suggestions")
                suggestions = solution['suggestions']

                # Show unserved summary
                unserved = suggestions.get('unserved_summary', {})
                if unserved:
                    st.warning(f"{unserved['count']} customers could not be served")

                    # Show reasons for unserved customers
                    if unserved.get('reasons', {}).get('capacity', 0) > 0:
                        st.error(f"🚚 {unserved['reasons']['capacity']} customers exceed vehicle capacity")
                    if unserved.get('reasons', {}).get('time', 0) > 0:
                        st.warning(f"⏱️ {unserved['reasons']['time']} customers couldn't be served within time constraints")

                # Display each suggestion
                for suggestion in suggestions.get('suggestions', []):
                    with st.expander(f"🔹 {suggestion['message']}"):
                        st.write(suggestion.get('suggestion', ''))
                        if 'details' in suggestion:
                            st.json(suggestion['details'], expanded=False)

            # Export data to CSV
            st.subheader("📤 Export Results")
            routes_data = []

            # Export Served Customers
            for route_idx, route in enumerate(solution['routes']):
                for stop_idx, node_idx in enumerate(route):
                    if node_idx == 0:
                        routes_data.append({
                            'Type': 'Warehouse', 'Status': 'Served',
                            'Route': route_idx + 1, 'Stop': stop_idx,
                            'Customer Name': 'Warehouse', 'Address': 'Warehouse',
                            'Customer ID': '0', 'Quantity': 0
                        })
                    else:
                        if st.session_state.data is not None and node_idx - 1 < len(st.session_state.data):
                            customer = st.session_state.data.iloc[node_idx - 1]
                            routes_data.append({
                                'Type': 'Customer', 'Status': 'Served',
                                'Route': route_idx + 1, 'Stop': stop_idx,
                                'Customer Name': customer.get('שם לקוח', ''),
                                'Address': customer.get('כתובת', ''),
                                'Customer ID': str(customer.get('מס\' לקוח', '')),
                                'Quantity': round(customer.get('avg_quantity', 0), 1),
                                'Reason': ''
                            })

            # Export Unserved Customers
            unserved_list = solution.get('unserved', [])
            for u in unserved_list:
                if st.session_state.data is not None and u['id'] - 1 < len(st.session_state.data):
                    customer = st.session_state.data.iloc[u['id'] - 1]
                    routes_data.append({
                        'Type': 'Customer', 'Status': 'Not Served',
                        'Route': 'N/A', 'Stop': 'N/A',
                        'Customer Name': customer.get('שם לקוח', ''),
                        'Address': customer.get('כתובת', ''),
                        'Customer ID': str(customer.get('מס\' לקוח', '')),
                        'Quantity': u['demand'],
                        'Reason': u.get('reason', 'Not served')
                    })

            # Display and export the data
            routes_df = pd.DataFrame(routes_data)

            # Convert columns with mixed types to object to prevent Arrow serialization errors
            if 'Route' in routes_df.columns:
                routes_df['Route'] = routes_df['Route'].astype(object)
            if 'Stop' in routes_df.columns:
                routes_df['Stop'] = routes_df['Stop'].astype(object)

            st.dataframe(routes_df, width="stretch")

            # Add download button
            csv = routes_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "💾 Download Full Report (CSV)",
                csv,
                "optimization_results.csv",
                "text/csv",
                key='download-csv'
            )

        except Exception as e:
            st.error(f"Export Error: {str(e)}")
    else:
        st.warning("No solution to export.")


def tab_compare_actuals(services: Optional[Dict[str, Any]]) -> None:
    """Handle the performance comparison tab functionality."""
    st.header("📊 Performance Comparison")
    
    # Handle case where services might be None (shouldn't happen after fix, but defensive)
    if services is None:
        st.warning("⚠️ Application configuration error. Please refresh the page.")
        return

    if not st.session_state.solution or not st.session_state.solution.get('solution_found'):
        st.warning("Please run an optimization first before comparing results.")
        return

    # Check if simulation data is available
    if not hasattr(st.session_state, 'distance_matrix') or not hasattr(st.session_state, 'time_matrix'):
        st.warning("⚠️ Simulation data not available. Please run optimization again to enable historical backtesting.")
        st.info("The optimization process caches distance/time matrices needed for simulation.")
        return

    # Check if main order data is available
    if st.session_state.data is None:
        st.warning("⚠️ Please upload order data in the Data Upload tab first.")
        return

    # Check if raw data is available (needed for date filtering)
    if not hasattr(st.session_state, 'raw_data') or st.session_state.raw_data is None:
        st.warning("⚠️ Raw order data is not available. Please re-upload your order file in the Data Upload tab to enable date filtering.")
        st.info("The raw data is needed to filter orders by date range. Please upload your file again.")
        return

    st.subheader("Upload Summary File for Date Range")
    st.info("ℹ️ Upload your summary file (e.g., Kogol T file) containing the date range. The simulation will use the order data from the Data Upload tab filtered by this date range.")
    
    summary_file = st.file_uploader("Upload Summary File (CSV/Excel)", type=['csv', 'xlsx', 'xls'], key='summary_file')

    if summary_file:
        try:
            # Load the summary file
            df_summary = pd.read_csv(summary_file) if summary_file.name.endswith('.csv') else pd.read_excel(summary_file)

            # Check for date range column
            date_range_col = 'טווח תאריכים'
            
            if date_range_col not in df_summary.columns:
                st.error(f"Summary file must contain a date range column: '{date_range_col}'")
                st.info(f"Available columns: {', '.join(df_summary.columns)}")
                return

            # Parse date range from first row (assuming single range for now)
            date_range_str = df_summary[date_range_col].iloc[0]
            
            if pd.isna(date_range_str):
                st.error(f"Date range column is empty in the summary file.")
                return

            # Parse date range (format: [dd.mm.yyyy-dd.mm.yyyy] or DD/MM/YYYY-DD/MM/YYYY)
            try:
                # Remove square brackets if present
                date_range_str = str(date_range_str).strip()
                if date_range_str.startswith('[') and date_range_str.endswith(']'):
                    date_range_str = date_range_str[1:-1].strip()
                
                date_parts = date_range_str.split('-')
                if len(date_parts) != 2:
                    raise ValueError("Date range must be in format [dd.mm.yyyy-dd.mm.yyyy] or DD/MM/YYYY-DD/MM/YYYY")
                
                start_date_str = date_parts[0].strip()
                end_date_str = date_parts[1].strip()
                
                # Replace dots with slashes if present (handle dd.mm.yyyy format)
                if '.' in start_date_str:
                    start_date_str = start_date_str.replace('.', '/')
                if '.' in end_date_str:
                    end_date_str = end_date_str.replace('.', '/')
                
                # Parse dates with dayfirst=True for DD/MM/YYYY format
                start_date = pd.to_datetime(start_date_str, dayfirst=True, errors='coerce')
                end_date = pd.to_datetime(end_date_str, dayfirst=True, errors='coerce')
                
                if pd.isna(start_date) or pd.isna(end_date):
                    raise ValueError(f"Could not parse dates: {date_parts[0]} - {date_parts[1]}")
                
                # Ensure end_date includes the full day
                end_date = end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                
            except Exception as date_e:
                st.error(f"Error parsing date range '{date_range_str}': {str(date_e)}")
                st.info("Expected format: [dd.mm.yyyy-dd.mm.yyyy] or DD/MM/YYYY-DD/MM/YYYY (e.g., [01.07.2025-31.07.2025] or 1/7/2025-31/7/2025)")
                return

            st.success(f"📅 Date range parsed: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")

            # Use raw data for date filtering (has individual order dates)
            raw_data = st.session_state.raw_data.copy()
            
            # Check if raw data has a date column
            order_date_col = 'תאריך אספקה'
            if order_date_col not in raw_data.columns:
                st.error(f"Raw order data must contain a date column: '{order_date_col}'")
                st.info(f"Available columns: {', '.join(raw_data.columns)}")
                return

            # Convert date column to datetime if needed
            if raw_data[order_date_col].dtype == 'object':
                raw_data[order_date_col] = pd.to_datetime(raw_data[order_date_col], dayfirst=True, errors='coerce')
            
            # Filter by date range
            filtered_data = raw_data[
                (raw_data[order_date_col] >= start_date) & 
                (raw_data[order_date_col] <= end_date)
            ].copy()

            if len(filtered_data) == 0:
                st.warning(f"⚠️ No orders found in the raw data for the date range {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
                st.info("Please check that the date column in your order data matches the date range format.")
                return

            st.success(f"✅ Filtered {len(filtered_data)} orders from raw data for the specified date range")

            # Merge coordinates from aggregated data back into filtered raw data
            # The aggregated data has lat/lng from geocoding
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
                
                # Count how many orders got coordinates
                coords_count = filtered_data[['lat', 'lng']].notna().all(axis=1).sum()
                if coords_count < len(filtered_data):
                    st.warning(f"⚠️ Only {coords_count} of {len(filtered_data)} orders have coordinates. Orders without coordinates will be excluded from simulation.")
            else:
                st.warning("⚠️ Could not merge coordinates. Customer ID column may be missing.")
                st.error("Cannot run simulation without coordinates. Please ensure data has been geocoded.")
                return

            # Filter out rows without coordinates (needed for simulation)
            filtered_data_with_coords = filtered_data[
                filtered_data[['lat', 'lng']].notna().all(axis=1)
            ].copy()

            if len(filtered_data_with_coords) == 0:
                st.error("⚠️ No orders with valid coordinates found after filtering. Please ensure your data has been geocoded.")
                return

            # Run simulation on filtered data with coordinates
            from calculations.simulation import run_historical_simulation
            
            with st.spinner("Running historical backtesting simulation (this may take a moment)..."):
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
            
            st.success(f"✅ Simulation complete: {len(simulation_results)} days analyzed")

            # Convert Date column to datetime for filtering
            simulation_results['Date'] = pd.to_datetime(simulation_results['Date'])

            # Add date filter
            st.subheader("📅 Date Range Selection")
            available_dates = sorted(simulation_results['Date'].dt.date.unique())
            selected_dates = st.multiselect(
                "Select dates to include in comparison",
                options=available_dates,
                default=available_dates,
                format_func=lambda x: x.strftime('%d/%m/%Y')
            )

            if not selected_dates:
                st.warning("⚠️ Please select at least one date for comparison.")
                return

            # Filter simulation results by selected dates
            selected_date_objs = [pd.Timestamp(d) for d in selected_dates]
            filtered_simulation = simulation_results[
                simulation_results['Date'].dt.date.isin(selected_dates)
            ].copy()

            num_selected_days = len(filtered_simulation)

            # Calculate summary file totals and daily averages
            summary_total_km = None
            summary_total_hours = None
            total_days_in_summary = (end_date - start_date).days + 1
            
            # Check for summary metrics columns
            if 'סהכ קמ' in df_summary.columns:
                summary_total_km = df_summary['סהכ קמ'].sum()
            elif 'Total KM' in df_summary.columns:
                summary_total_km = df_summary['Total KM'].sum()
            
            if 'סהכ זמן נסיעה' in df_summary.columns:
                summary_total_hours = df_summary['סהכ זמן נסיעה'].sum()
            elif 'Total Driving Time' in df_summary.columns:
                summary_total_hours = df_summary['Total Driving Time'].sum()

            # Calculate daily averages from summary file
            avg_actual_km_per_day = None
            avg_actual_time_per_day = None
            if summary_total_km is not None and total_days_in_summary > 0:
                avg_actual_km_per_day = summary_total_km / total_days_in_summary
            if summary_total_hours is not None and total_days_in_summary > 0:
                avg_actual_time_per_day = summary_total_hours / total_days_in_summary

            # Calculate projected and simulated totals for selected days
            projected_actual_km = None
            projected_actual_hours = None
            if avg_actual_km_per_day is not None:
                projected_actual_km = avg_actual_km_per_day * num_selected_days
            if avg_actual_time_per_day is not None:
                projected_actual_hours = avg_actual_time_per_day * num_selected_days

            simulated_km = filtered_simulation['Total_Distance_km'].sum()
            simulated_hours = filtered_simulation['Max_Shift_Duration_hours'].sum()

            # Display comparison table
            if projected_actual_km is not None or projected_actual_hours is not None:
                st.subheader("💰 Comparison: Simulated vs. Projected Actuals")
                
                comparison_data = []
                if projected_actual_km is not None:
                    km_diff = simulated_km - projected_actual_km
                    km_diff_pct = (km_diff / projected_actual_km * 100) if projected_actual_km > 0 else 0
                    comparison_data.append({
                        'Metric': 'Total Distance (km)',
                        'Projected Actuals (Based on Avg)': f"{projected_actual_km:,.1f}",
                        'Simulated': f"{simulated_km:,.1f}",
                        'Difference': f"{km_diff:,.1f}",
                        'Difference %': f"{km_diff_pct:.2f}%"
                    })
                if projected_actual_hours is not None:
                    hours_diff = simulated_hours - projected_actual_hours
                    hours_diff_pct = (hours_diff / projected_actual_hours * 100) if projected_actual_hours > 0 else 0
                    comparison_data.append({
                        'Metric': 'Total Time (hours)',
                        'Projected Actuals (Based on Avg)': f"{projected_actual_hours:,.1f}",
                        'Simulated': f"{simulated_hours:,.1f}",
                        'Difference': f"{hours_diff:,.1f}",
                        'Difference %': f"{hours_diff_pct:.2f}%"
                    })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    st.dataframe(comp_df.set_index('Metric'), width="stretch")
                    
                    # Show savings message
                    if projected_actual_km is not None:
                        km_savings = projected_actual_km - simulated_km
                        km_savings_pct = (km_savings / projected_actual_km * 100) if projected_actual_km > 0 else 0
                        if km_savings > 0:
                            st.success(f"✅ Simulated routes would save {km_savings:,.1f} km ({km_savings_pct:.1f}%) compared to projected actuals")
                        elif km_savings < 0:
                            st.warning(f"⚠️ Simulated routes would use {abs(km_savings):,.1f} km more ({abs(km_savings_pct):.1f}%) than projected actuals")

            # Fleet Utilization Comparison
            st.subheader("🚛 Fleet Utilization Comparison")
            
            # System metric: Average Active_Routes from simulation
            system_avg_trucks = filtered_simulation['Active_Routes'].mean()
            
            # Historical metric: Count unique route names per day from filtered raw data
            historical_avg_trucks = None
            route_name_col = 'שם קו'
            
            if route_name_col in filtered_data.columns:
                # Filter historical data to selected dates
                filtered_data['Date'] = pd.to_datetime(filtered_data[order_date_col], dayfirst=True, errors='coerce')
                historical_for_selected = filtered_data[
                    filtered_data['Date'].dt.date.isin(selected_dates)
                ].copy()
                
                if len(historical_for_selected) > 0:
                    # Group by date and count unique route names
                    daily_route_counts = historical_for_selected.groupby(
                        historical_for_selected['Date'].dt.date
                    )[route_name_col].nunique()
                    
                    if len(daily_route_counts) > 0:
                        historical_avg_trucks = daily_route_counts.mean()
            
            # Display fleet utilization metrics
            if historical_avg_trucks is not None:
                fleet_diff = system_avg_trucks - historical_avg_trucks
                
                col1, col2, col3 = st.columns(3)
                col1.metric("System Avg Trucks", f"{system_avg_trucks:.1f}")
                col2.metric("Historical Avg Trucks", f"{historical_avg_trucks:.1f}")
                col3.metric("Difference", f"{fleet_diff:+.1f} trucks", 
                           delta=f"{fleet_diff:+.1f}" if abs(fleet_diff) > 0.1 else None)
            else:
                st.info(f"System Average Trucks: {system_avg_trucks:.1f}")
                st.warning("⚠️ Historical route data ('שם קו') not available for comparison.")

            # Summary metrics
            st.subheader("📈 Summary Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Selected Days", num_selected_days)
            col2.metric("Avg Daily Distance", f"{filtered_simulation['Total_Distance_km'].mean():.1f} km")
            col3.metric("Avg Shift Duration", f"{filtered_simulation['Max_Shift_Duration_hours'].mean():.1f} hrs")

            # Check for overtime days
            max_shift_hours = st.session_state.optimization_params['max_shift_seconds'] / 3600
            overtime_days = filtered_simulation[filtered_simulation['Max_Shift_Duration_hours'] > max_shift_hours]
            if len(overtime_days) > 0:
                st.warning(f"⚠️ {len(overtime_days)} of {num_selected_days} selected days would have exceeded the maximum shift time ({max_shift_hours} hours)")

            # Variance customers warning
            total_variance_customers = filtered_simulation['Variance_Customers'].sum()
            if total_variance_customers > 0:
                st.info(f"ℹ️ {total_variance_customers} customer orders in selected period were not in the optimized routes (new/unknown customers).")

            # Download button for filtered results
            csv = filtered_simulation.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "💾 Download Filtered Simulation Results (CSV)",
                csv,
                "filtered_simulation_results.csv",
                "text/csv",
                key='download-filtered-simulation-csv'
            )

        except Exception as e:
            st.error(f"Error processing historical data: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return
