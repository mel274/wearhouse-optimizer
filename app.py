import streamlit as st
import os
import pandas as pd
import numpy as np
from config import Config
import utils
from data_manager import DataManager
from geo_service import GeoService
# Updated Import: Use the new modular optimizer
from calculations.route_optimizer import RouteOptimizer
from visualizer import MapBuilder
from session_manager import SessionManager
from exceptions import WarehouseOptimizerError, DataValidationError, GeocodingError, OptimizationError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session manager
session_manager = SessionManager()

def init_session_state():
    if 'data' not in st.session_state: st.session_state.data = None
    if 'solution' not in st.session_state: st.session_state.solution = None
    if 'geo_service' not in st.session_state: st.session_state.geo_service = None
    if 'warehouse_coords' not in st.session_state: st.session_state.warehouse_coords = None
    if 'file_id' not in st.session_state: st.session_state.file_id = None
    if 'operation_mode' not in st.session_state: st.session_state.operation_mode = "Strategic Planning (Center of Gravity)"
    if 'last_operation_mode' not in st.session_state: st.session_state.last_operation_mode = st.session_state.operation_mode

def setup_sidebar():
    if os.path.exists("michelin-logo-1.png"):
        st.sidebar.image("michelin-logo-1.png", use_container_width=True)
    st.sidebar.header("⚙️ Settings")

    # --- Mode Selector ---
    modes = ["Strategic Planning (Center of Gravity)", "Daily Route Planning"]
    st.session_state.operation_mode = st.sidebar.radio(
        "**Operation Mode**", modes, 
        help="'Strategic' finds the best warehouse location. 'Daily' plans routes from a fixed warehouse."
    )

    # --- State Cleanup on Mode Change ---
    if st.session_state.operation_mode != st.session_state.get('last_operation_mode'):
        keys_to_clear = ['warehouse_coords', 'solution', 'data', 'file_id']
        for key in keys_to_clear:
            if key in st.session_state:
                st.session_state[key] = None
        st.session_state.last_operation_mode = st.session_state.operation_mode
        st.info("Mode changed. Cleared session state. Rerunning...")
        st.rerun()
    
    # Initialize services
    if Config.OPENROUTESERVICE_API_KEY or Config.RADAR_API_KEY:
        if st.session_state.geo_service is None:
            # GeoService now strictly handles Geocoding
            st.session_state.geo_service = GeoService(Config.OPENROUTESERVICE_API_KEY)
        st.sidebar.success("✅ API Key loaded")
    else:
        st.sidebar.error("⚠️ Missing API Key")
        return None
    
    st.sidebar.subheader("Fleet Parameters")
    fleet_size = st.sidebar.slider("Max Fleet Size (Limit)", 1, 50, Config.DEFAULT_FLEET_SIZE, 
                                 help="The solver will try to use fewer trucks if possible.")
    truck_capacity = st.sidebar.number_input("Truck Capacity", 100, 10000, Config.DEFAULT_CAPACITY, 100)
    max_shift_hours = st.sidebar.slider("Max Shift (Hours)", 1, 24, Config.MAX_SHIFT_HOURS)
    service_time_minutes = st.sidebar.slider("Service Time (Min)", 5, 60, Config.SERVICE_TIME_MINUTES)
    
    return {
        'fleet_size': fleet_size,
        'truck_capacity': truck_capacity,
        'max_shift_hours': max_shift_hours,
        'service_time_minutes': service_time_minutes,
        'data_manager': DataManager(),
        'map_builder': MapBuilder(),
        # Instantiate new modular optimizer
        'route_optimizer': RouteOptimizer(Config.OPENROUTESERVICE_API_KEY)
    }

def tab_data_upload(services):
    st.header("📊 Data Upload")
    if services is None: return
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
                        geocoded_data = services['data_manager'].add_coordinates(st.session_state.data, st.session_state.geo_service)
                        st.session_state.data = geocoded_data
                        st.rerun()

                if geocoded_count > 0:
                    warehouse_location_for_map = None

                    # --- Daily Mode: Manual Warehouse Input ---
                    if st.session_state.operation_mode == "Daily Route Planning":
                        st.subheader("📍 Set Warehouse Location")
                        manual_address = st.text_input("Enter Warehouse Address", "")
                        if manual_address and st.button("Geocode Warehouse Address"):
                            try:
                                coords = st.session_state.geo_service.get_coordinates(manual_address)
                                if coords:
                                    st.session_state.warehouse_coords = coords
                                    st.success(f"Warehouse geocoded to: {coords}")
                                    st.rerun()
                                else:
                                    st.error("Could not geocode address.")
                            except Exception as e:
                                st.error(f"Geocoding failed: {e}")
                        
                        if st.session_state.warehouse_coords:
                            lat, lng = st.session_state.warehouse_coords
                            warehouse_location_for_map = {'lat': lat, 'lng': lng}

                    # --- Strategic Mode: Center of Gravity ---
                    else:
                        st.subheader("📈 Center of Gravity Analysis")
                        cog = services['data_manager'].calculate_center_of_gravity(st.session_state.data)
                        
                        if cog['lat'] and st.session_state.geo_service:
                            raw_coords = (cog['lat'], cog['lng'])
                            snapped_coords = st.session_state.geo_service.snap_to_road(raw_coords)
                            
                            if snapped_coords != raw_coords:
                                cog['lat'], cog['lng'] = snapped_coords
                                st.info(f"ℹ️ Warehouse Auto-Snapped to nearest road: {snapped_coords}")

                        st.session_state.warehouse_coords = (cog['lat'], cog['lng'])
                        warehouse_location_for_map = cog

                    # --- Map Visualization (Common to both modes) ---
                    if warehouse_location_for_map and warehouse_location_for_map['lat'] is not None:
                        st.subheader("🗺️ Customer & Warehouse Map")
                        phase1_map = services['map_builder'].create_phase1_map(st.session_state.data, warehouse_location_for_map)
                        st.components.v1.html(phase1_map._repr_html_(), height=600)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def tab_optimization(services):
    st.header("🚛 Route Optimization")
    if services is None or st.session_state.data is None:
        st.warning("Upload data first.")
        return
    
    geocoded_data = st.session_state.data
    valid_coords = geocoded_data.dropna(subset=['lat', 'lng'])
    if valid_coords.empty:
        st.error("No geocoded data.")
        return
    
    # Use the snapped coords from Tab 1, or recalculate if missing
    if st.session_state.warehouse_coords:
        default_wh = f"{st.session_state.warehouse_coords[0]:.6f}, {st.session_state.warehouse_coords[1]:.6f}"
    else:
        cog = services['data_manager'].calculate_center_of_gravity(valid_coords)
        default_wh = f"{cog['lat']:.6f}, {cog['lng']:.6f}" if cog['lat'] else ""

    warehouse_address = st.text_input("Warehouse Location (Snapped)", value=default_wh, help="This is the starting point for all trucks.")
    
    if st.session_state.operation_mode == "Daily Route Planning":
        demand_col = 'total_quantity'
        st.info("Using 'Total Quantity' for daily demand.")
    else:
        demand_col = 'avg_quantity'
        st.info("Using 'Average Quantity' for strategic planning.")
    
    if demand_col not in valid_coords.columns:
        st.error(f"Demand column '{demand_col}' not found in the uploaded data.")
        return
        
    estimated_demand = np.ceil(valid_coords[demand_col]).astype(int)
    
    st.info(f"ℹ️ **Strategy:** \n1. **K-Means**: Initial Grouping.\n2. **Matrix Refinement**: Swap customers to correct routes.\n3. **Route Merging**: Combine routes to save stem kilometers.\n4. **Optimization**: Final detailed routing.")
    
    if st.button("🚀 Run Auto-Optimization", type="primary"):
        try:
            with st.spinner("Running Optimization Stages..."):
                # Parse Warehouse Coords
                if ',' in warehouse_address:
                    try: wh_coords = tuple(map(float, warehouse_address.split(',')))
                    except: wh_coords = st.session_state.geo_service.get_coordinates(warehouse_address)
                else: wh_coords = st.session_state.geo_service.get_coordinates(warehouse_address)
                
                st.session_state.warehouse_coords = wh_coords
                
                matrix_coords = [wh_coords] + list(zip(valid_coords['lat'], valid_coords['lng']))
                demands = [0] + estimated_demand.tolist()
                
                params = {
                    'fleet_size': services['fleet_size'],
                    'capacity': services['truck_capacity'],
                    'max_shift_seconds': services['max_shift_hours'] * 3600,
                    'service_time_seconds': services['service_time_minutes'] * 60
                }
                
                # CALL THE NEW OPTIMIZER
                solution = services['route_optimizer'].solve(
                    [f"Loc {i}" for i in range(len(matrix_coords))],
                    demands,
                    params,
                    coords=matrix_coords
                )
                
                st.session_state.solution = solution
                
                if solution['solution_found']:
                    num_routes = len(solution['routes'])
                    st.success(f"Optimization Complete: {num_routes} trucks used")
                    
                    # --- MAP Visualization ---
                    st.subheader("🗺️ Route Map")
                    # Pass geo_service explicitly for polylines if needed by visualizer, 
                    # though optimizer now returns polylines in metrics.
                    phase2_map = services['map_builder'].create_phase2_map(
                        solution, valid_coords, wh_coords, 
                        geo_service=st.session_state.geo_service
                    )
                    st.components.v1.html(phase2_map._repr_html_(), height=700)

                    # --- Expandable Route Details ---
                    st.subheader("📋 Route Details")
                    
                    route_metrics = solution.get('route_metrics', [])
                    
                    for idx, route in enumerate(solution['routes']):
                        if len(route) <= 2: continue
                        
                        current_load = sum(demands[node] for node in route)
                        r_metrics = route_metrics[idx] if idx < len(route_metrics) else {'distance': 0, 'duration': 0}
                        dist_km = r_metrics.get('distance', 0) / 1000
                        dur_str = utils.format_time(r_metrics.get('duration', 0))
                        
                        warnings = []
                        if current_load > services['truck_capacity']: warnings.append("⚠️ Overloaded")
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

                    # --- Global Metrics ---
                    m = solution['metrics']
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Distance", f"{m['total_distance']/1000:.1f} km")
                    c2.metric("Total Time", f"{m['total_time']/3600:.1f} h")
                    c3.metric("Vehicles Used", m['num_vehicles_used'])

                else:
                    st.error("Optimization failed.")
                    
        except Exception as e:
            st.error(f"Optimization Error: {str(e)}")
            logger.error(f"Optimization error: {e}")

def tab_export():
    st.header("📤 Export Results")
    if st.session_state.solution and st.session_state.solution['solution_found']:
        try:
            routes_data = []
            solution = st.session_state.solution
            
            for route_idx, route in enumerate(solution['routes']):
                for stop_idx, node_idx in enumerate(route):
                    if node_idx == 0:
                        routes_data.append({
                            'Route': route_idx + 1, 'Stop': stop_idx, 'Type': 'Warehouse',
                            'Customer Name': 'Warehouse', 'Address': 'Warehouse', 'Customer ID': '0',
                            'Quantity (Avg)': 0
                        })
                    else:
                        if st.session_state.data is not None and node_idx - 1 < len(st.session_state.data):
                            customer = st.session_state.data.iloc[node_idx - 1]
                            routes_data.append({
                                'Route': route_idx + 1, 
                                'Stop': stop_idx, 
                                'Type': 'Customer',
                                'Customer Name': customer.get('שם לקוח', ''),
                                'Address': customer.get('כתובת', ''),
                                'Customer ID': str(customer.get('מס\' לקוח', '')),
                                'Quantity (Avg)': round(customer.get('avg_quantity', 0), 1),
                                'Visits': customer.get('num_visits', 0)
                            })
            
            routes_df = pd.DataFrame(routes_data)
            st.subheader("Optimized Routes")
            st.dataframe(routes_df, width="stretch")
            csv = routes_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("📥 Download CSV", csv, "optimized_routes.csv", "text/csv")
            
        except Exception as e:
            st.error(f"Export Error: {str(e)}")
    else:
        st.warning("No solution to export.")

def main():
    st.set_page_config(layout="wide", page_title="Warehouse Optimizer", page_icon="🚛")
    init_session_state()
    services = setup_sidebar()
    st.title("🏭 Warehouse Location & Route Optimizer")
    tab1, tab2, tab3 = st.tabs(["1. Data Upload", "2. Optimization", "3. Export"])
    with tab1: tab_data_upload(services)
    with tab2: tab_optimization(services)
    with tab3: tab_export()

if __name__ == "__main__":
    main()