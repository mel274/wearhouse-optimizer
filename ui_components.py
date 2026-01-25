"""
UI components and setup functions for the Warehouse Optimizer Streamlit app.
"""
import streamlit as st
import os
from typing import Dict, Any, Optional
from config import Config
from geo_service import GeoService
from calculations.fleet_service import FleetService
from calculations.comparison_service import ComparisonService


def setup_sidebar() -> Dict[str, Any]:
    """
    Setup and configure the sidebar with settings and parameters.

    Returns:
        Dictionary with services and parameters. Services may be None if API keys are missing,
        but the dictionary structure is always returned to support graceful degradation.
    """
    if os.path.exists("michelin-logo-1.png"):
        st.sidebar.image("michelin-logo-1.png")
    st.sidebar.header("⚙️ Settings")

    st.sidebar.subheader("Warehouse Location")
    warehouse_address = st.sidebar.text_input(
        "Manual Warehouse Address",
        "",
        help="Enter a full address to override any auto-calculation."
    )

    # Initialize services
    if Config.OPENROUTESERVICE_API_KEY or Config.RADAR_API_KEY:
        if st.session_state.geo_service is None:
            st.session_state.geo_service = GeoService(Config.OPENROUTESERVICE_API_KEY)
    else:
        st.sidebar.warning("⚠️ API Key not configured. Geocoding and optimization features will be disabled.")
        st.session_state.geo_service = None

    st.sidebar.subheader("Fleet Parameters")

    # Fleet Composition
    st.sidebar.subheader("Fleet Composition")
    fleet_cols = st.sidebar.columns(2)

    with fleet_cols[0]:
        num_big_trucks = st.number_input("Big Trucks", 1, 12, 12, key="big_trucks")

    with fleet_cols[1]:
        num_small_trucks = st.number_input("Small Trucks", 1, 6, 6, key="small_trucks")
    # Max Shift Duration Selector
    st.sidebar.subheader("Max Shift Duration")
    duration_cols = st.sidebar.columns([1, 1])

    with duration_cols[0]:
        hours = st.number_input("Hours", 1, 24, Config.MAX_SHIFT_HOURS, key="shift_hours")

    with duration_cols[1]:
        minutes_options = [0, 10, 20, 30, 40, 50]
        minutes = st.selectbox("Minutes", minutes_options, index=0, format_func=lambda x: f"{x:02d} min", key="shift_minutes")

    max_shift_hours = hours + (minutes / 60.0)
    service_time_minutes = st.sidebar.number_input("Service Time (Minutes)", 1, 60, Config.SERVICE_TIME_MINUTES)

    st.sidebar.subheader("Volume & Logistics")
    big_truck_vol = st.sidebar.number_input(
        "Big Truck Volume (m³)", 1.0, 100.0, Config.FLEET_DEFAULTS['big_truck_vol'], 0.5
    )
    small_truck_vol = st.sidebar.number_input(
        "Small Truck Volume (m³)", 1.0, 100.0, Config.FLEET_DEFAULTS['small_truck_vol'], 0.5
    )
    safety_factor = st.sidebar.slider("Safety Factor", 0.1, 1.0, Config.FLEET_DEFAULTS['safety_factor'], 0.05)
    safety_buffer = st.sidebar.slider(
        'Buffer',
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Statistical multiplier for Standard Deviation (Force Calculation)"
    )
    
    st.sidebar.subheader("Exception Tolerance")
    available_exception_percent = st.sidebar.slider(
        'Available Exception %',
        min_value=0,
        max_value=10,
        value=int(Config.AVAILABLE_EXCEPTION_PERCENT * 100),
        step=1,
        format="%d%%",
        help="Maximum percentage of total drives that can exceed limits. If exceeded, system will retry with additional Big Trucks."
    ) / 100.0  # Convert back to decimal


    return {
        'warehouse_address': warehouse_address,
        'num_big_trucks': num_big_trucks,
        'num_small_trucks': num_small_trucks,
        'max_shift_hours': max_shift_hours,
        'service_time_minutes': service_time_minutes,
        'available_exception_percent': available_exception_percent,
        'data_manager': None,  # Will be imported when needed
        'map_builder': None,   # Will be imported when needed
        'route_optimizer': None,  # Will be imported when needed
        'fleet_service': FleetService(),
        'comparison_service': ComparisonService(),
        'fleet_settings': {
            'big_truck_vol': big_truck_vol,
            'small_truck_vol': small_truck_vol,
            'safety_factor': safety_factor,
            'safety_buffer': safety_buffer
        }
    }


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'solution' not in st.session_state:
        st.session_state.solution = None
    if 'geo_service' not in st.session_state:
        st.session_state.geo_service = None
    if 'warehouse_coords' not in st.session_state:
        st.session_state.warehouse_coords = None
    if 'file_id' not in st.session_state:
        st.session_state.file_id = None
    # Raw data (preserved before aggregation for date filtering)
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    # Simulation data (for historical backtesting)
    if 'distance_matrix' not in st.session_state:
        st.session_state.distance_matrix = None
    if 'time_matrix' not in st.session_state:
        st.session_state.time_matrix = None
    if 'node_map' not in st.session_state:
        st.session_state.node_map = None
    if 'optimization_params' not in st.session_state:
        st.session_state.optimization_params = None
    if 'valid_coords_for_simulation' not in st.session_state:
        st.session_state.valid_coords_for_simulation = None