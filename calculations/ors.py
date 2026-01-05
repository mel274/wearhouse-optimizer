"""
OpenRouteService API interactions with enhanced timeout and retry logic.
"""
import requests
import logging
import math
import time
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .types import MatrixData, Coords, RouteMetrics
from .utils import decode_polyline
from utils import retry_with_backoff

logger = logging.getLogger(__name__)

# Default timeouts in seconds
DEFAULT_CONNECT_TIMEOUT = 10
DEFAULT_READ_TIMEOUT = 120  # Increased from 30 to 120 seconds
MAX_RETRIES = 3

def _compute_matrix_cache_key(locations: List[Coords]) -> str:
    """Compute a stable hash key for the given coordinates."""
    # Create a stable representation of coordinates
    coords_str = repr(sorted(locations))
    return hashlib.sha1(coords_str.encode()).hexdigest()[:16]

class ORSHandler:
    def __init__(self, api_key: str, connect_timeout: int = DEFAULT_CONNECT_TIMEOUT, 
                 read_timeout: int = DEFAULT_READ_TIMEOUT, max_retries: int = MAX_RETRIES):
        """
        Initialize the ORS handler with configurable timeouts and retries.
        
        Args:
            api_key: OpenRouteService API key
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
            max_retries: Maximum number of retries for failed requests
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
        Fetch distance and duration matrix with enhanced error handling and disk caching.
        
        Args:
            locations: List of (lat, lng) coordinates
            timeout: Optional timeout in seconds (can be tuple of (connect_timeout, read_timeout))
            
        Returns:
            Dictionary containing 'durations' and 'distances' matrices
            
        Raises:
            ValueError: If API key is missing or locations list is empty
            requests.exceptions.RequestException: For request-related errors
        """
        if not self.api_key:
            raise ValueError("API Key is required for distance matrix requests.")
            
        if not locations:
            raise ValueError("At least one location is required.")
        
        # Check cache first
        cache_key = _compute_matrix_cache_key(locations)
        cache_dir = Path(".cache/matrices")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"matrix_{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                logger.info(f"Loading distance matrix from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Successfully loaded cached matrix for {len(locations)} locations")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}. Fetching from API.")
        
        logger.info(f"Fetching distance matrix for {len(locations)} locations")
        
        # Use instance timeout if not specified
        timeout = timeout or self.timeout
        
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
            matrix_data = {
                'durations': data['durations'],
                'distances': data['distances']
            }
            
            # Save to cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(matrix_data, f)
                logger.info(f"Cached distance matrix to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save cache file {cache_file}: {e}")
            
            return matrix_data
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timed out after {timeout} seconds: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise

    @retry_with_backoff(max_attempts=3)
    def get_directions(self, route_coordinates: List[Coords]) -> Dict[str, Any]:
        """
        Fetch route geometry (polylines) and precise distance/duration from ORS Directions API.
        
        Args:
            route_coordinates: List of (lat, lng) coordinates in stop order
            
        Returns:
            Dictionary with 'geometry' (list of (lat, lng) tuples), 'distance' (meters), and 'duration' (seconds)
            
        Raises:
            ValueError: If API key is missing or coordinates list is empty
            requests.exceptions.RequestException: For request-related errors
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
                # Convert [lon, lat] -> [lat, lon]
                polylines.append([(lat, lng) for lng, lat in route_data['geometry']['coordinates']])
        return polylines

    def get_fallback_geometry(self, start: Coords, end: Coords) -> List[Coords]:
        """Get simple geometry or call directions API if needed (fallback)."""
        # For strict modularity, we might just return straight line here 
        # or implement a simple directions call. Returning straight line for robustness.
        return [start, end]