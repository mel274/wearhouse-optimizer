"""
OpenRouteService API interactions.
"""
import requests
import logging
import math
from typing import List, Dict, Any, Tuple
from .types import MatrixData, Coords, RouteMetrics
from .utils import retry_with_backoff, decode_polyline

logger = logging.getLogger(__name__)

class ORSHandler:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not self.api_key:
            logger.warning("ORSHandler initialized without API Key.")

    @retry_with_backoff(max_attempts=3)
    def get_distance_matrix(self, locations: List[Coords]) -> MatrixData:
        """Fetch distance and duration matrix."""
        if not self.api_key:
            raise ValueError("API Key missing.")

        # ORS expects [lon, lat]
        formatted_locations = [[loc[1], loc[0]] for loc in locations]
        
        url = "https://api.openrouteservice.org/v2/matrix/driving-car"
        headers = {
            'Authorization': self.api_key,
            'Content-Type': 'application/json'
        }
        body = {
            "locations": formatted_locations,
            "metrics": ["distance", "duration"]
        }

        response = requests.post(url, json=body, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        return {
            'durations': data['durations'],
            'distances': data['distances']
        }

    @retry_with_backoff(max_attempts=2)
    def optimize_route(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """Calls the ORS VRP Optimization endpoint."""
        url = "https://api.openrouteservice.org/optimization"
        headers = {
            'Authorization': self.api_key,
            'Content-Type': 'application/json'
        }
        response = requests.post(url, json=request_body, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_route_polylines(self, route_data: Dict, start_coords: Coords) -> List[List[Coords]]:
        """Extracts polylines from optimization response."""
        polylines = []
        if 'geometry' in route_data:
            if isinstance(route_data['geometry'], str):
                decoded = decode_polyline(route_data['geometry'])
                if decoded: polylines.append(decoded)
            elif isinstance(route_data['geometry'], dict) and 'coordinates' in route_data['geometry']:
                # Convert [lon, lat] -> [lat, lon]
                polylines.append([(lat, lng) for lng, lat in route_data['geometry']['coordinates']])
        return polylines

    def get_fallback_geometry(self, start: Coords, end: Coords) -> List[Coords]:
        """Get simple geometry or call directions API if needed (fallback)."""
        # For strict modularity, we might just return straight line here 
        # or implement a simple directions call. Returning straight line for robustness.
        return [start, end]