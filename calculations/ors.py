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

# Matrix API chunk size - ORS API limit is ~50 locations per request
# Using 40 to stay safely under the limit
MATRIX_CHUNK_SIZE = 40

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
        Fetch distance and duration matrix with enhanced error handling, disk caching, and batching.
        
        For large location sets (>40), this method automatically batches requests into smaller chunks
        to avoid API limits, then reassembles the full matrix.
        
        Args:
            locations: List of (lat, lng) coordinates
            timeout: Optional timeout in seconds (can be tuple of (connect_timeout, read_timeout))
            
        Returns:
            Dictionary containing 'durations' and 'distances' matrices (N x N)
            
        Raises:
            ValueError: If API key is missing or locations list is empty
            requests.exceptions.RequestException: For request-related errors
        """
        if not self.api_key:
            raise ValueError("API Key is required for distance matrix requests.")
            
        if not locations:
            raise ValueError("At least one location is required.")
        
        num_locations = len(locations)
        
        # Check cache first (cache key is based on all locations)
        cache_key = _compute_matrix_cache_key(locations)
        cache_dir = Path(".cache/matrices")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"matrix_{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                logger.info(f"Loading distance matrix from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Successfully loaded cached matrix for {num_locations} locations")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}. Fetching from API.")
        
        # Use instance timeout if not specified
        timeout = timeout or self.timeout
        
        # If locations count is small enough, use direct API call (faster)
        if num_locations <= MATRIX_CHUNK_SIZE:
            logger.info(f"Fetching distance matrix for {num_locations} locations (single request)")
            return self._fetch_matrix_direct(locations, timeout, cache_file)
        
        # For large location sets, use batched approach
        logger.info(f"Fetching distance matrix for {num_locations} locations (batched into chunks of {MATRIX_CHUNK_SIZE})")
        return self._fetch_matrix_batched(locations, timeout, cache_file)
    
    def _fetch_matrix_direct(self, locations: List[Coords], timeout: Tuple[float, float], 
                            cache_file: Path) -> MatrixData:
        """Fetch matrix directly for small location sets (<= MATRIX_CHUNK_SIZE)."""
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
    
    def _fetch_matrix_batched(self, locations: List[Coords], timeout: Tuple[float, float],
                             cache_file: Path) -> MatrixData:
        """
        Fetch matrix using batched requests for large location sets.
        
        Breaks the request into chunks of MATRIX_CHUNK_SIZE and reassembles the full matrix.
        """
        num_locations = len(locations)
        
        # Step 1: Create full-sized matrices (N x N) initialized with zeros
        durations_matrix = [[0.0] * num_locations for _ in range(num_locations)]
        distances_matrix = [[0.0] * num_locations for _ in range(num_locations)]
        
        # Step 2: Loop through source chunks
        url = "https://api.openrouteservice.org/v2/matrix/driving-car"
        headers = {
            'Authorization': self.api_key,
            'Content-Type': 'application/json'
        }
        
        total_chunks = ((num_locations + MATRIX_CHUNK_SIZE - 1) // MATRIX_CHUNK_SIZE) ** 2
        chunk_count = 0
        
        for source_start in range(0, num_locations, MATRIX_CHUNK_SIZE):
            source_end = min(source_start + MATRIX_CHUNK_SIZE, num_locations)
            source_chunk = locations[source_start:source_end]
            source_formatted = [[loc[1], loc[0]] for loc in source_chunk]
            
            # Step 3: Loop through destination chunks
            for dest_start in range(0, num_locations, MATRIX_CHUNK_SIZE):
                dest_end = min(dest_start + MATRIX_CHUNK_SIZE, num_locations)
                dest_chunk = locations[dest_start:dest_end]
                dest_formatted = [[loc[1], loc[0]] for loc in dest_chunk]
                
                chunk_count += 1
                logger.info(f"Fetching chunk {chunk_count}/{total_chunks}: sources [{source_start}:{source_end}], destinations [{dest_start}:{dest_end}]")
                
                # Small delay between requests to avoid rate limiting (except for first request)
                if chunk_count > 1:
                    time.sleep(0.5)  # 500ms delay between chunk requests
                
                # Step 4: Fetch sub-matrix for this chunk pair
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
                    
                    # Step 5: Stitch the sub-matrix into the correct positions in main matrices
                    # chunk_durations[i][j] corresponds to source_chunk[i] -> dest_chunk[j]
                    # source_chunk[i] is locations[source_start + i]
                    # dest_chunk[j] is locations[dest_start + j]
                    # So we map: durations_matrix[source_start + i][dest_start + j] = chunk_durations[i][j]
                    for i in range(len(source_chunk)):
                        for j in range(len(dest_chunk)):
                            if i < len(chunk_durations) and j < len(chunk_durations[i]):
                                source_idx = source_start + i
                                dest_idx = dest_start + j
                                durations_matrix[source_idx][dest_idx] = chunk_durations[i][j]
                                distances_matrix[source_idx][dest_idx] = chunk_distances[i][j]
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to fetch chunk [{source_start}:{source_end}] x [{dest_start}:{dest_end}]: {str(e)}")
                    raise
        
        # Step 6: Return the fully populated matrices
        matrix_data = {
            'durations': durations_matrix,
            'distances': distances_matrix
        }
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(matrix_data, f)
            logger.info(f"Cached batched distance matrix to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")
        
        logger.info(f"Successfully assembled full matrix ({num_locations} x {num_locations}) from {chunk_count} chunks")
        return matrix_data

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