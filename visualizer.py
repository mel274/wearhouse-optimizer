"""
Map visualization service for warehouse optimization and route planning.
Version: 2.0 (Failed Customer Support)
"""

import folium
import folium.plugins
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MapBuilder:
    """Service for creating interactive maps with Folium."""
    
    def __init__(self):
        """Initialize MapBuilder."""
        self.colors = ['red', 'blue', 'green', 'purple', 'orange', 
                      'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
                      'cadetblue', 'pink', 'gray', 'black']
    
    def create_phase1_map(self, df: pd.DataFrame, cog: Optional[Dict[str, float]]) -> folium.Map:
        """
        Create Phase 1 map showing customers and warehouse location (if provided).
        Customers are shown as small fixed dots.
        
        Args:
            df: DataFrame with customer data including lat/lng columns
            cog: Optional dictionary with 'lat' and 'lng' keys for warehouse location.
                 If None, no warehouse marker will be displayed.
        """
        try:
            # Center map on warehouse or first customer
            if cog is not None and cog.get('lat') is not None and cog.get('lng') is not None:
                center_lat, center_lng = cog['lat'], cog['lng']
            else:
                # Fallback to first customer with valid coordinates
                valid_coords = df.dropna(subset=['lat', 'lng'])
                if not valid_coords.empty:
                    center_lat = valid_coords.iloc[0]['lat']
                    center_lng = valid_coords.iloc[0]['lng']
                else:
                    center_lat, center_lng = 31.7683, 35.2137  # Default to Jerusalem
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lng],
                zoom_start=11,
                tiles='OpenStreetMap'
            )
            
            # Add warehouse marker only if coordinates are provided
            if cog is not None and cog.get('lat') is not None and cog.get('lng') is not None:
                folium.Marker(
                    location=[cog['lat'], cog['lng']],
                    popup='<div>Warehouse Location</div>',
                    icon=folium.Icon(color='red', icon='star', prefix='fa')
                ).add_to(m)
            
            # Add customer markers
            for idx, row in df.iterrows():
                if pd.notna(row['lat']) and pd.notna(row['lng']):
                    radius = 3 
                    
                    popup_html = f"""
                    <div style="font-family: Arial, sans-serif; font-size: 12px;">
                        <strong>{row.get('שם לקוח', 'N/A')}</strong><br>
                        Address: {row.get('כתובת', 'N/A')}<br>
                        Volume: {round(row.get('force_volume', 0), 2)} m³
                    </div>
                    """
                    
                    marker_color = row.get('color', 'blue') # Default to blue if no color column

                    folium.CircleMarker(
                        location=[row['lat'], row['lng']],
                        radius=radius,
                        popup=folium.Popup(popup_html, max_width=300),
                        color=marker_color,
                        fill=True,
                        fillColor=marker_color,
                        fillOpacity=0.8,
                        weight=1
                    ).add_to(m)
            
            folium.LayerControl().add_to(m)
            folium.plugins.Fullscreen().add_to(m)
            
            return m
            
        except Exception as e:
            logger.error(f"Error creating Phase 1 map: {e}")
            raise
    
    def create_phase2_map(self, solution: Dict[str, Any], locations_df: pd.DataFrame, 
                         warehouse_coords: Tuple[float, float], geo_service=None) -> folium.Map:
        """
        Create Phase 2 map showing optimized delivery routes and unserved customers.
        """
        try:
            if not solution.get('solution_found', False):
                raise ValueError("No solution found to visualize")
            
            # Center map on warehouse
            center_lat, center_lng = warehouse_coords
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lng],
                zoom_start=12,
                tiles='OpenStreetMap'
            )
            
            # Add warehouse marker
            folium.Marker(
                location=[center_lat, center_lng],
                popup='<div><strong>Warehouse</strong></div>',
                icon=folium.Icon(color='darkred', icon='home', prefix='fa')
            ).add_to(m)
            
            # 1. Process Valid Routes
            routes = solution.get('routes', [])
            route_metrics = solution.get('route_metrics', [])
            
            for route_idx, route in enumerate(routes):
                if len(route) <= 1: continue
                
                color = self.colors[route_idx % len(self.colors)]
                
                # Check for pre-calculated polylines in route metrics
                route_polylines = None
                if route_idx < len(route_metrics):
                    metrics = route_metrics[route_idx]
                    polylines = metrics.get('polylines', [])
                    if polylines and len(polylines) > 0:
                        # polylines is a list of coordinate lists, use the first one (full route)
                        route_polylines = polylines[0]
                
                # Add stop markers for all stops in the route
                stop_counter = 0  # Track customer stops (excluding depot)
                for i in range(len(route)):
                    curr_node_idx = route[i]
                    if curr_node_idx == 0:
                        # Skip warehouse marker (already added at the beginning)
                        continue
                    else:
                        if curr_node_idx - 1 < len(locations_df):
                            cust = locations_df.iloc[curr_node_idx - 1]
                            curr_coords = (cust['lat'], cust['lng'])
                            # Increment stop counter (first customer = stop 1, second = stop 2, etc.)
                            stop_counter += 1
                            self._add_stop_marker(m, cust, curr_coords, stop_counter, route_idx + 1, color)
                
                # Draw route path using pre-calculated polylines if available
                if route_polylines:
                    # Use pre-calculated geometry from optimization
                    folium.PolyLine(
                        locations=route_polylines,
                        color=color,
                        weight=4,
                        opacity=0.8,
                        popup=f'<div>Route {route_idx + 1}</div>'
                    ).add_to(m)
                else:
                    # Fallback: build path segment by segment
                    full_route_path = []
                    for i in range(len(route)):
                        curr_node_idx = route[i]
                        if curr_node_idx == 0:
                            curr_coords = warehouse_coords
                        else:
                            if curr_node_idx - 1 < len(locations_df):
                                cust = locations_df.iloc[curr_node_idx - 1]
                                curr_coords = (cust['lat'], cust['lng'])
                            else:
                                continue
                        
                        # Connect to next node
                        if i < len(route) - 1:
                            next_node_idx = route[i+1]
                            if next_node_idx == 0:
                                next_coords = warehouse_coords
                            else:
                                if next_node_idx - 1 < len(locations_df):
                                    cust_next = locations_df.iloc[next_node_idx - 1]
                                    next_coords = (cust_next['lat'], cust_next['lng'])
                                else:
                                    continue

                            # Try to get polyline from geo_service if available
                            if geo_service:
                                try:
                                    segment_points = geo_service.get_route_polyline(curr_coords, next_coords)
                                    if segment_points:
                                        full_route_path.extend(segment_points)
                                    else:
                                        full_route_path.extend([curr_coords, next_coords])
                                except Exception:
                                    # Fallback to straight line if geo_service fails
                                    full_route_path.extend([curr_coords, next_coords])
                            else:
                                # No geo_service, use straight line
                                full_route_path.extend([curr_coords, next_coords])
                    
                    # Draw polyline for route (fallback method)
                    if full_route_path:
                        folium.PolyLine(
                            locations=full_route_path,
                            color=color,
                            weight=4,
                            opacity=0.8,
                            popup=f'<div>Route {route_idx + 1}</div>'
                        ).add_to(m)

            # 2. Process Unserved Customers (Failed)
            unserved_list = solution.get('unserved', [])
            for unserved in unserved_list:
                u_idx = unserved['id']
                if u_idx - 1 < len(locations_df):
                    cust = locations_df.iloc[u_idx - 1]
                    coords = (cust['lat'], cust['lng'])
                    reason = unserved.get('reason', 'Unknown error')
                    
                    self._add_failed_marker(m, cust, coords, reason)

            folium.LayerControl().add_to(m)
            folium.plugins.Fullscreen().add_to(m)
            
            return m
            
        except Exception as e:
            logger.error(f"Error creating Phase 2 map: {e}")
            raise

    def _add_stop_marker(self, m, customer, coords, stop_num, route_num, color):
        """Helper to add a numbered marker for a stop."""
        popup_html = f"""
        <div style="font-family: Arial, sans-serif;">
            <strong>{customer.get('שם לקוח', 'N/A')}</strong><br>
            Stop: {stop_num}<br>
            Route: {route_num}<br>
            Volume: {round(customer.get('force_volume', 0), 2)} m³
        </div>
        """
        
        folium.Marker(
            location=coords,
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.DivIcon(
                html=f'<div style="background-color: {color}; color: white; '
                     f'border-radius: 50%; width: 24px; height: 24px; '
                     f'display: flex; align-items: center; justify-content: center; '
                     f'font-weight: bold; font-size: 11px; border: 2px solid white; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">{stop_num}</div>',
                icon_size=(24, 24),
                icon_anchor=(12, 12)
            )
        ).add_to(m)

    def _add_failed_marker(self, m, customer, coords, reason):
        """Helper to add a warning marker for failed customers."""
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; color: #a94442;">
            <strong>{customer.get('שם לקוח', 'N/A')}</strong><br>
            <span style="font-weight: bold;">⚠️ COULD NOT SERVE</span><br>
            Reason: {reason}<br>
            Volume: {round(customer.get('force_volume', 0), 2)} m³
        </div>
        """
        
        folium.Marker(
            location=coords,
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa'),
            tooltip=f"Unserved: {customer.get('שם לקוח', 'N/A')}"
        ).add_to(m)