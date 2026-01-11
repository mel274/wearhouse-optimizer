"""
OpenRouteService API interactions with enhanced timeout and retry logic.
Caching has been DISABLED for Distance Matrix and Directions to ensure data freshness.
"""
import requests
import logging
import math
import time
from typing import List, Dict, Any, Tuple, Optional, Union
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .types import MatrixData, Coords, RouteMetrics
from shared.utils import decode_polyline, retry_with_backoff
from config import Config

logger = logging.getLogger(__name__)

class ORSHandler:
    def __init__(self, api_key: str, connect_timeout: int = Config.ORS_CONNECT_TIMEOUT,
                 read_timeout: int = Config.ORS_READ_TIMEOUT, max_retries: int = Config.ORS_MAX_RETRIES):
        """
        Initialize the ORS handler with configurable timeouts and retries.
        """
        self.api_key = api_key
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.timeout = (connect_timeout, read_timeout)
        self.max_retries = max_retries
        
        # Configure session with retry strategy
        self.session = self._create_session()
        
        if not self.api_key:
            logger.warning("ORSHandler initialized without API Key.")
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @retry_with_backoff(max_attempts=3)
    def get_distance_matrix(self, locations: List[Coords], timeout: Optional[Union[float, Tuple[float, float]]] = None) -> MatrixData:
        """
        Fetch distance and duration matrix.
        NO CACHING performed. Always fetches fresh data from API.
        """
        if not self.api_key:
            raise ValueError("API Key is required for distance matrix requests.")
            
        if not locations:
            raise ValueError("At least one location is required.")
        
        num_locations = len(locations)
        
        # Use instance timeout if not specified
        timeout = timeout or self.timeout
        
        # If locations count is small enough, use direct API call (faster)
        if num_locations <= Config.ORS_MATRIX_CHUNK_SIZE:
            logger.info(f"Fetching distance matrix for {num_locations} locations (single request)")
            return self._fetch_matrix_direct(locations, timeout)
        
        # For large location sets, use batched approach
        logger.info(f"Fetching distance matrix for {num_locations} locations (batched into chunks of {Config.ORS_MATRIX_CHUNK_SIZE})")
        return self._fetch_matrix_batched(locations, timeout)
    
    def _fetch_matrix_direct(self, locations: List[Coords], timeout: Tuple[float, float]) -> MatrixData:
        """Fetch matrix directly for small location sets."""
        try:
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

            response = self.session.post(
                url, 
                json=body, 
                headers=headers, 
                timeout=timeout
            )
            response.raise_for_status()
            
            data = response.json()
            return {
                'durations': data['durations'],
                'distances': data['distances']
            }
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timed out after {timeout} seconds: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise
    
    def _fetch_matrix_batched(self, locations: List[Coords], timeout: Tuple[float, float]) -> MatrixData:
        """
        Fetch matrix using batched requests for large location sets.
        """
        num_locations = len(locations)
        
        # Create full-sized matrices (N x N) initialized with zeros
        durations_matrix = [[0.0] * num_locations for _ in range(num_locations)]
        distances_matrix = [[0.0] * num_locations for _ in range(num_locations)]
        
        # Loop through source chunks
        url = "https://api.openrouteservice.org/v2/matrix/driving-car"
        headers = {
            'Authorization': self.api_key,
            'Content-Type': 'application/json'
        }
        
        total_chunks = ((num_locations + Config.ORS_MATRIX_CHUNK_SIZE - 1) // Config.ORS_MATRIX_CHUNK_SIZE) ** 2
        chunk_count = 0
        
        for source_start in range(0, num_locations, Config.ORS_MATRIX_CHUNK_SIZE):
            source_end = min(source_start + Config.ORS_MATRIX_CHUNK_SIZE, num_locations)
            source_chunk = locations[source_start:source_end]
            source_formatted = [[loc[1], loc[0]] for loc in source_chunk]
            
            # Loop through destination chunks
            for dest_start in range(0, num_locations, Config.ORS_MATRIX_CHUNK_SIZE):
                dest_end = min(dest_start + Config.ORS_MATRIX_CHUNK_SIZE, num_locations)
                dest_chunk = locations[dest_start:dest_end]
                dest_formatted = [[loc[1], loc[0]] for loc in dest_chunk]
                
                chunk_count += 1
                logger.info(f"Fetching chunk {chunk_count}/{total_chunks}")
                
                # Small delay between requests to avoid rate limiting
                if chunk_count > 1:
                    time.sleep(0.5)
                
                try:
                    body = {
                        "locations": source_formatted + dest_formatted,
                        "sources": list(range(len(source_formatted))),
                        "destinations": list(range(len(source_formatted), len(source_formatted) + len(dest_formatted))),
                        "metrics": ["distance", "duration"]
                    }
                    
                    response = self.session.post(
                        url,
                        json=body,
                        headers=headers,
                        timeout=timeout
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    chunk_durations = data['durations']
                    chunk_distances = data['distances']
                    
                    # Fill the main matrix
                    for i in range(len(source_chunk)):
                        for j in range(len(dest_chunk)):
                            if i < len(chunk_durations) and j < len(chunk_durations[i]):
                                source_idx = source_start + i
                                dest_idx = dest_start + j
                                durations_matrix[source_idx][dest_idx] = chunk_durations[i][j]
                                distances_matrix[source_idx][dest_idx] = chunk_distances[i][j]
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to fetch chunk: {str(e)}")
                    raise
        
        return {
            'durations': durations_matrix,
            'distances': distances_matrix
        }

    @retry_with_backoff(max_attempts=3)
    def get_directions(self, route_coordinates: List[Coords]) -> Dict[str, Any]:
        """
        Fetch route geometry (polylines) and precise distance/duration from ORS Directions API.
        """
        if not self.api_key:
            raise ValueError("API Key is required for directions requests.")
            
        if not route_coordinates or len(route_coordinates) < 2:
            raise ValueError("At least two coordinates are required for directions.")
        
        logger.info(f"Fetching directions for route with {len(route_coordinates)} waypoints")
        
        try:
            # ORS expects [lon, lat]
            formatted_coords = [[coord[1], coord[0]] for coord in route_coordinates]
            
            url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
            headers = {
                'Authorization': self.api_key,
                'Content-Type': 'application/json'
            }
            body = {
                "coordinates": formatted_coords
            }
            
            response = self.session.post(
                url,
                json=body,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract geometry from GeoJSON
            geometry = []
            if 'features' in data and len(data['features']) > 0:
                feature = data['features'][0]
                if 'geometry' in feature and 'coordinates' in feature['geometry']:
                    # Convert [lon, lat] -> (lat, lon)
                    geometry = [(coord[1], coord[0]) for coord in feature['geometry']['coordinates']]
            
            # Extract distance and duration from properties
            distance = 0.0
            duration = 0.0
            if 'features' in data and len(data['features']) > 0:
                properties = data['features'][0].get('properties', {})
                summary = properties.get('summary', {})
                distance = summary.get('distance', 0.0)  # meters
                duration = summary.get('duration', 0.0)  # seconds
            
            return {
                'geometry': geometry,
                'distance': distance,
                'duration': duration
            }
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Directions request timed out: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Directions request failed: {str(e)}")
            raise

    def get_route_polylines(self, route_data: Dict, start_coords: Coords) -> List[List[Coords]]:
        """Extracts polylines from optimization response."""
        polylines = []
        if 'geometry' in route_data:
            if isinstance(route_data['geometry'], str):
                decoded = decode_polyline(route_data['geometry'])
                if decoded: polylines.append(decoded)
            elif isinstance(route_data['geometry'], dict) and 'coordinates' in route_data['geometry']:
                polylines.append([(lat, lng) for lng, lat in route_data['geometry']['coordinates']])
        return polylines

    def get_fallback_geometry(self, start: Coords, end: Coords) -> List[Coords]:
        """Get simple geometry or call directions API if needed (fallback)."""
        return [start, end]