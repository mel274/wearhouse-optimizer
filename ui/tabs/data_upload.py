"""
Data upload tab for the Warehouse Optimizer Streamlit app.
Handles file upload, data processing, and geocoding functionality.
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from data_manager import DataManager
from calculations.logistics import calculate_fleet_metrics
from visualizer import MapBuilder
import shared.utils as utils


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
