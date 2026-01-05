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
    if services is None:
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
            if st.session_state.file_id != file_id:
                raw_data = services['data_manager'].load_data(uploaded_file)
                aggregated_data = services['data_manager'].aggregate_data(raw_data)
                st.session_state.data = aggregated_data
                st.session_state.file_id = file_id
                st.session_state.solution = None
                st.session_state.warehouse_coords = None
                st.success(f"Loaded {len(aggregated_data)} customers")

            if st.session_state.data is not None:
                st.dataframe(st.session_state.data)
                geocoded_count = st.session_state.data[['lat', 'lng']].notna().all(axis=1).sum()

                if st.session_state.geo_service and st.button("📍 Geocode Addresses", type="primary"):
                    with st.spinner("Geocoding..."):
                        geocoded_data = services['data_manager'].add_coordinates(
                            st.session_state.data, st.session_state.geo_service
                        )
                        st.session_state.data = geocoded_data
                        st.rerun()

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
    if services is None or st.session_state.data is None:
        st.warning("Upload data first.")
        return

    # Initialize route optimizer if not already done
    if services['route_optimizer'] is None:
        from config import Config
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

    if not st.session_state.solution or not st.session_state.solution.get('solution_found'):
        st.warning("Please run an optimization first before comparing results.")
        return

    st.subheader("Upload Actuals Report")
    actuals_file = st.file_uploader("Upload your historical data (CSV/Excel)", type=['csv', 'xlsx', 'xls'])

    if actuals_file:
        try:
            # Load the file
            df_actuals = pd.read_csv(actuals_file) if actuals_file.name.endswith('.csv') else pd.read_excel(actuals_file)

            # Define Hebrew to English column mapping
            hebrew_to_english = {
                'טווח תאריכים': 'Date Range',
                'שם קו': 'Line Name',
                'סהכ קמ': 'Total KM',
                'סהכ זמן נסיעה': 'Total Driving Time'
            }

            # Check for required Hebrew columns
            required_hebrew_cols = list(hebrew_to_english.keys())
            if not all(col in df_actuals.columns for col in required_hebrew_cols):
                st.error(f"Invalid file format. Please ensure the file contains the columns: {', '.join(required_hebrew_cols)}")
                return

            # Rename columns to English for processing
            df_actuals = df_actuals.rename(columns=hebrew_to_english)

            st.success("Actuals report uploaded successfully.")

            # Data Processing Logic
            # Parse Date Range
            try:
                date_range_str = df_actuals['Date Range'].iloc[0]
                start_date_str, end_date_str = date_range_str.split('-')
                # Using dayfirst=True to correctly parse DD/MM/YYYY
                start_date = pd.to_datetime(start_date_str, dayfirst=True)
                end_date = pd.to_datetime(end_date_str, dayfirst=True)
                num_days = (end_date - start_date).days + 1
            except Exception as date_e:
                st.error(f"Could not parse the 'Date Range' column. Please ensure it is in 'DD/MM/YYYY-DD/MM/YYYY' format. Error: {date_e}")
                return

            # Aggregate Actuals
            actual_total_km = df_actuals['Total KM'].sum()
            actual_total_hours = df_actuals['Total Driving Time'].sum()
            actual_trucks_used = df_actuals['Line Name'].nunique()

            # Aggregate Optimized Results (Projected)
            optimized_metrics = st.session_state.solution['metrics']
            service_time_per_stop_hours = services['service_time_minutes'] / 60
            total_stops = sum(len(r) - 2 for r in st.session_state.solution['routes'])

            # Calculate total driving time (from optimizer) and add total service time
            optimized_driving_hours_single_day = optimized_metrics.get('total_time', 0) / 3600
            optimized_service_time_single_day = total_stops * service_time_per_stop_hours
            optimized_total_hours_single_day = optimized_driving_hours_single_day + optimized_service_time_single_day

            # Project results over the period
            projected_optimized_km = (optimized_metrics.get('total_distance', 0) / 1000) * num_days
            projected_optimized_hours = optimized_total_hours_single_day * num_days
            optimized_trucks_used = optimized_metrics.get('num_vehicles_used', 0)

            st.info(f"Comparing actuals from **{start_date.strftime('%d %b %Y')}** to **{end_date.strftime('%d %b %Y')}** ({num_days} days) against projected optimization results.")

            # Visualization
            st.subheader("💰 Savings Scorecard")
            mileage_savings = actual_total_km - projected_optimized_km
            hours_savings = actual_total_hours - projected_optimized_hours
            trucks_savings = actual_trucks_used - optimized_trucks_used

            mileage_savings_percent = (mileage_savings / actual_total_km) * 100 if actual_total_km > 0 else 0
            hours_savings_percent = (hours_savings / actual_total_hours) * 100 if actual_total_hours > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Mileage Saved", f"{mileage_savings:,.1f} km", f"{mileage_savings_percent:.1f}%", delta_color="inverse")
            col2.metric("Hours Saved", f"{hours_savings:,.1f} hrs", f"{hours_savings_percent:.1f}%", delta_color="inverse")
            col3.metric("Fleet Reduction", f"{trucks_savings} Trucks", delta_color="inverse")

            # Charts
            st.subheader("📊 Visual Comparison")
            chart_data = pd.DataFrame({
                'Category': ['Actual', 'Optimized'],
                'Total KM': [actual_total_km, projected_optimized_km],
                'Total Hours': [actual_total_hours, projected_optimized_hours]
            })

            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(chart_data.set_index('Category')['Total KM'], use_container_width=True)
            with col2:
                st.bar_chart(chart_data.set_index('Category')['Total Hours'], use_container_width=True)

            # Data Table
            st.subheader("📋 Summary Table")
            summary_df = pd.DataFrame({
                'Metric': ['Total Mileage (km)', 'Total Driving Hours', 'Asset Utilization (Trucks)'],
                'Actual': [f"{actual_total_km:,.1f}", f"{actual_total_hours:,.1f}", actual_trucks_used],
                'Optimized': [f"{projected_optimized_km:,.1f}", f"{projected_optimized_hours:,.1f}", optimized_trucks_used]
            })
            st.table(summary_df.set_index('Metric'))

        except Exception as e:
            st.error(f"Error reading or processing the actuals file: {e}")
            return
