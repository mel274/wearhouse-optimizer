"""
Static Routes tab for the Warehouse Optimizer Streamlit app.
Displays permanent routes with a simplified view (map and route list).
"""
import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any


def tab_static_routes(services: Optional[Dict[str, Any]]) -> None:
    """Handle the static routes tab functionality."""
    # Check for Solution
    if st.session_state.get('solution') is None:
        st.warning("No optimization solution found. Please run optimization first.")
        return
    
    # Get Data
    solution = st.session_state.solution
    valid_coords = st.session_state.valid_coords_for_simulation
    wh_coords = st.session_state.warehouse_coords
    
    # Map Section
    st.header("Permanent Routes Map")
    phase2_map = services['map_builder'].create_phase2_map(
        solution, valid_coords, wh_coords,
        geo_service=st.session_state.geo_service
    )
    st.components.v1.html(phase2_map._repr_html_(), height=700)
    
    # Routes List Section
    st.header("Route Assignments")
    
    route_metrics = solution.get('route_metrics', [])
    
    for idx, route in enumerate(solution['routes']):
        if len(route) <= 2:  # Skip empty routes
            continue
        
        # Get vehicle type from route metrics
        if idx < len(route_metrics):
            vehicle_type = route_metrics[idx].get('vehicle_type', 'Big')
        else:
            vehicle_type = 'Big'
        
        # Simplified Header
        header = f"Truck {idx+1} ({vehicle_type})"
        
        with st.expander(header):
            # Simplified Content: Create list of stops
            stops_data = []
            for stop_order, node_idx in enumerate(route[1:-1]):  # Skip depot (first and last nodes)
                if node_idx > 0:  # Skip depot node
                    customer = valid_coords.iloc[node_idx - 1]
                    stops_data.append({
                        "Stop #": stop_order + 1,
                        "Name": customer.get("שם לקוח", "")
                    })
            
            if stops_data:
                st.dataframe(pd.DataFrame(stops_data), use_container_width=True, hide_index=True)
            else:
                st.write("Warehouse return only.")
