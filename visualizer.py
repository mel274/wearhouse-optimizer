"""
Map visualization service for warehouse optimization and route planning.
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
    
    def create_phase1_map(self, df: pd.DataFrame, cog: Dict[str, float]) -> folium.Map:
        """
        Create Phase 1 map showing customers and center of gravity.
        Customers are shown as small fixed dots.
        """
        try:
            # Center map on center of gravity or first customer
            if cog['lat'] and cog['lng']:
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
            
            # Add center of gravity marker
            if cog['lat'] and cog['lng']:
                folium.Marker(
                    location=[cog['lat'], cog['lng']],
                    popup='<div>Center of Gravity</div>',
                    icon=folium.Icon(color='red', icon='star', prefix='fa')
                ).add_to(m)
                
                # Optional: Keep the influence area but make it very subtle or remove if requested.
                # Keeping it for now as context, but customers are now just dots.
                folium.Circle(
                    location=[cog['lat'], cog['lng']],
                    radius=2000,  # 2km radius
                    color='red',
                    fill=True,
                    fillOpacity=0.05,
                    weight=1,
                    popup='<div>Influence Area (2 km)</div>'
                ).add_to(m)
            
            # Add customer markers
            for idx, row in df.iterrows():
                if pd.notna(row['lat']) and pd.notna(row['lng']):
                    # UPDATED: Fixed radius for all customers (Just a dot)
                    radius = 3 
                    
                    # Create popup
                    popup_html = f"""
                    <div style="font-family: Arial, sans-serif; font-size: 12px;">
                        <strong>{row.get('שם לקוח', 'N/A')}</strong><br>
                        Address: {row.get('כתובת', 'N/A')}<br>
                        Avg Qty: {round(row.get('avg_quantity', 0), 1)}
                    </div>
                    """
                    
                    folium.CircleMarker(
                        location=[row['lat'], row['lng']],
                        radius=radius,
                        popup=folium.Popup(popup_html, max_width=300),
                        color='blue',       # Border color
                        fill=True,
                        fillColor='blue',   # Fill color
                        fillOpacity=0.8,
                        weight=1
                    ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)

            folium.plugins.Fullscreen(
                position='topright',
                title='Expand map',
                title_cancel='Exit full screen',
                force_separate_button=True
            ).add_to(m)
            
            logger.info("Phase 1 map created successfully")
            return m
            
        except Exception as e:
            logger.error(f"Error creating Phase 1 map: {e}")
            raise
    
    def create_phase2_map(self, solution: Dict[str, Any], locations_df: pd.DataFrame, 
                         warehouse_coords: Tuple[float, float], geo_service=None) -> folium.Map:
        """
        Create Phase 2 map showing optimized delivery routes on actual roads.
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
            
            # Process routes
            routes = solution.get('routes', [])
            
            for route_idx, route in enumerate(routes):
                if len(route) <= 1:  # Skip empty routes
                    continue
                
                # Get color for this route
                color = self.colors[route_idx % len(self.colors)]
                
                # Collect coordinates for this route
                full_route_path = []
                
                for i in range(len(route)):
                    # Get current node coordinates
                    curr_node_idx = route[i]
                    if curr_node_idx == 0:
                        curr_coords = warehouse_coords
                    else:
                        # Get customer info (adjust for 0-based depot)
                        if curr_node_idx - 1 < len(locations_df):
                            cust = locations_df.iloc[curr_node_idx - 1]
                            curr_coords = (cust['lat'], cust['lng'])
                            
                            # Add marker for the stop
                            self._add_stop_marker(m, cust, curr_coords, i, route_idx + 1, color)
                    
                    # Get path to next node
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

                        # Fetch Geometry
                        if geo_service:
                            segment_points = geo_service.get_route_polyline(curr_coords, next_coords)
                            full_route_path.extend(segment_points)
                        else:
                            full_route_path.extend([curr_coords, next_coords])
                
                # Draw route polyline
                if full_route_path:
                    folium.PolyLine(
                        locations=full_route_path,
                        color=color,
                        weight=4,
                        opacity=0.8,
                        popup=f'<div>Route {route_idx + 1}</div>'
                    ).add_to(m)
            
            # Add route layer control
            folium.LayerControl().add_to(m)

            folium.plugins.Fullscreen(
                position='topright',
                title='Expand map',
                title_cancel='Exit full screen',
                force_separate_button=True
            ).add_to(m)
            
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
            Avg Qty: {round(customer.get('avg_quantity', 0), 1)}
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