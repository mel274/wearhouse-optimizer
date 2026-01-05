"""
UI components and setup functions for the Warehouse Optimizer Streamlit app.
"""
import streamlit as st
import os
from typing import Dict, Any, Optional
from config import Config
from geo_service import GeoService


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
    fleet_size = st.sidebar.slider(
        "Max Fleet Size (12 Big + 6 Small)",
        1, 50, Config.DEFAULT_FLEET_SIZE,
        help="The solver will try to use fewer trucks if possible."
    )
    max_shift_hours = st.sidebar.slider("Max Shift (Hours)", 1, 24, Config.MAX_SHIFT_HOURS)
    service_time_minutes = st.sidebar.slider("Service Time (Min)", 5, 60, Config.SERVICE_TIME_MINUTES)

    st.sidebar.subheader("Volume & Logistics")
    big_truck_vol = st.sidebar.number_input(
        "Big Truck Volume (m³)", 1.0, 100.0, Config.FLEET_DEFAULTS['big_truck_vol'], 0.5
    )
    small_truck_vol = st.sidebar.number_input(
        "Small Truck Volume (m³)", 1.0, 100.0, Config.FLEET_DEFAULTS['small_truck_vol'], 0.5
    )
    safety_factor = st.sidebar.slider("Safety Factor", 0.1, 1.0, Config.FLEET_DEFAULTS['safety_factor'], 0.05)

    return {
        'warehouse_address': warehouse_address,
        'fleet_size': fleet_size,
        'max_shift_hours': max_shift_hours,
        'service_time_minutes': service_time_minutes,
        'data_manager': None,  # Will be imported when needed
        'map_builder': None,   # Will be imported when needed
        'route_optimizer': None,  # Will be imported when needed
        'fleet_settings': {
            'big_truck_vol': big_truck_vol,
            'small_truck_vol': small_truck_vol,
            'safety_factor': safety_factor
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