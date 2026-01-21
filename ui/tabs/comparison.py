"""
Comparison tab for the Warehouse Optimizer Streamlit app.
Handles performance comparison between optimized routes and historical actuals.
"""
import streamlit as st
import pandas as pd
from datetime import date
from typing import Dict, Any, Optional
from calculations.simulation import run_historical_simulation


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
    
    # Check if optimization params are available
    if not hasattr(st.session_state, 'optimization_params') or st.session_state.optimization_params is None:
        st.warning("Optimization parameters not available. Please run optimization again.")
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

    # Calculate date range
    min_date = raw_data[order_date_col].min().date()
    max_date = raw_data[order_date_col].max().date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        return

    filtered_data = raw_data[
        (raw_data[order_date_col].dt.date >= start_date) &
        (raw_data[order_date_col].dt.date <= end_date)
    ].copy()

    if len(filtered_data) == 0:
        st.warning(f"No orders found in the selected date range.")
        return

    # Parse summary file if provided
    df_summary = None
    summary_total_km = None
    summary_total_hours = None
    total_days_in_summary = None

    if summary_file:
        try:
            # Use ComparisonService to parse the summary file
            df_summary, summary_total_km, summary_total_hours, total_days_in_summary = \
                services['comparison_service'].parse_summary_file(summary_file)
        except Exception as e:
            st.warning(f"Could not load summary file: {str(e)}")

    # Prepare simulation data using ComparisonService
    try:
        filtered_data_with_coords = services['comparison_service'].prepare_simulation_data(
            st.session_state.raw_data, st.session_state.data, start_date, end_date
        )
    except ValueError as e:
        st.error(str(e))
        return

    # Run simulation
    with st.spinner("Running simulation..."):
        # Build route capacities list based on vehicle type
        solution = st.session_state.solution
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
        
        simulation_results = run_historical_simulation(
            historical_data=filtered_data_with_coords,
            master_routes=solution['routes'],
            distance_matrix=st.session_state.distance_matrix,
            time_matrix=st.session_state.time_matrix,
            node_map=st.session_state.node_map,
            service_time_seconds=st.session_state.optimization_params.get('service_time_seconds', 900),
            date_col=order_date_col,
            customer_id_col="מס' לקוח",
            depot=0,
            route_capacities=route_capacities,
            max_shift_seconds=st.session_state.optimization_params.get('max_shift_seconds'),
            volume_tolerance=services['fleet_settings'].get('volume_tolerance', 0.0),
            time_tolerance=services['fleet_settings'].get('time_tolerance', 0.0)
        )

    # Convert Date column to datetime for filtering
    simulation_results['Date'] = pd.to_datetime(simulation_results['Date'])

    # Use ComparisonService to calculate comparison metrics
    comparison_metrics = services['comparison_service'].calculate_comparison_metrics(
        simulation_results, filtered_data, summary_total_km, summary_total_hours,
        total_days_in_summary, (end_date - start_date).days + 1, start_date, end_date
    )

    # Display results
    num_selected_days = comparison_metrics['num_selected_days']

    # Fleet Utilization Comparison
    st.subheader("Fleet Utilization")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("System Avg Trucks", f"{comparison_metrics['system_avg_trucks']:.1f}")
    with col2:
        if comparison_metrics['historical_avg_trucks'] is not None:
            st.metric("Historical Avg Trucks", f"{comparison_metrics['historical_avg_trucks']:.1f}")
        else:
            st.metric("Historical Avg Trucks", "N/A")

    # Comparison Table
    st.subheader("Comparison: Projected vs. Optimized")

    if comparison_metrics['comparison_data']:
        comp_df = pd.DataFrame(comparison_metrics['comparison_data'])
        st.dataframe(comp_df.set_index('Metric'), width="stretch")
