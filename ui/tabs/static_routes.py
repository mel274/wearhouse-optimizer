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
    
    # Get fleet settings for capacity calculations
    safety_factor = services['fleet_settings']['safety_factor']
    small_capacity = services['fleet_settings']['small_truck_vol']
    big_capacity = services['fleet_settings']['big_truck_vol']
    
    for idx, route in enumerate(solution['routes']):
        if len(route) <= 2:  # Skip empty routes
            continue
        
        # Get vehicle type from route metrics
        if idx < len(route_metrics):
            vehicle_type = route_metrics[idx].get('vehicle_type', 'Big')
            distance_m = route_metrics[idx].get('distance', 0)
            duration_s = route_metrics[idx].get('duration', 0)
        else:
            vehicle_type = 'Big'
            distance_m = 0
            duration_s = 0
        
        # 1. Number of Stops (exclude depot)
        num_stops = len(route) - 2
        
        # 2. Load and 5. Volume (Effective) - calculate from customer data
        total_load = 0
        effective_vol = 0.0
        for node_idx in route[1:-1]:  # Skip depot (first and last nodes)
            if node_idx > 0:
                customer = valid_coords.iloc[node_idx - 1]
                force_volume = customer.get('force_volume', 0)
                eff_volume = force_volume / safety_factor
                effective_vol += eff_volume
                total_load += int(eff_volume * 1000)
        
        # 3. Distance in km
        distance_km = distance_m / 1000
        
        # 4. Time in HH:MM:SS format
        hours = int(duration_s // 3600)
        minutes = int((duration_s % 3600) // 60)
        seconds = int(duration_s % 60)
        time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # 5. Capacity based on vehicle type
        capacity = small_capacity if vehicle_type == 'Small' else big_capacity
        percentage = (effective_vol / capacity * 100) if capacity > 0 else 0
        
        # Build the detailed header
        header = (
            f"Truck {idx+1} ({vehicle_type}): {num_stops} Stops | "
            f"Load: {total_load} | Dist: {distance_km:.1f} km | "
            f"Time: {time_formatted} | Vol (Eff): {effective_vol:.1f} / {capacity:.1f} m³ "
            f"({percentage:.0f}%) ✅ OK"
        )
        
        with st.expander(header):
            # Content: Create list of stops
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
