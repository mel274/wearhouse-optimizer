"""
OpenRouteService API interactions with enhanced timeout and retry logic.
Caching has been DISABLED for Distance Matrix and Directions to ensure data freshness.
"""
import requests
import logging
import math
import time
import re
from typing import List, Dict, Any, Tuple, Optional, Union
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .types import MatrixData, Coords, RouteMetrics
from shared.utils import decode_polyline, retry_with_backoff
from config import Config

logger = logging.getLogger(__name__)


def _parse_bad_coordinate_index(error_message: str) -> Optional[int]:
    """
    Parse the ORS error message to extract the index of the bad coordinate.
    
    ORS errors look like: "Could not find routable point within a radius of 350.0 meters 
    of specified coordinate 3: 35.1846411 30.6140945"
    
    Returns the coordinate index (e.g., 3) or None if parsing fails.
    """
    try:
        # Match pattern like "coordinate 3:" or "coordinate 25:"
        match = re.search(r'coordinate\s+(\d+):', error_message)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None


class ORSHandler:
    # Maximum waypoints per directions request to avoid HTTP 400 errors
    MAX_WAYPOINTS_PER_REQUEST = 40

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

        # Hybrid architecture: Matrix operations use local Docker, Directions use cloud API
        if hasattr(Config, 'ORS_BASE_URL') and Config.ORS_BASE_URL:
            self.matrix_url = Config.ORS_BASE_URL.rstrip('/')  # Local Docker for Matrix operations
            logger.info(f"Matrix operations using local Docker ORS instance: {self.matrix_url}")
        else:
            self.matrix_url = "https://api.openrouteservice.org"
            logger.warning("No local Docker configured - Matrix operations will use cloud API")

        # Directions always use cloud API for better geometry support
        self.directions_url = "https://api.openrouteservice.org"
        logger.info(f"Directions operations using cloud ORS API: {self.directions_url}")

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
            
            url = f"{self.matrix_url}/matrix/driving-car"
            headers = {
                'Authorization': self.api_key,
                'Content-Type': 'application/json'
            }
            body = {
                "locations": formatted_locations,
                "metrics": ["distance", "duration"]
            }

            logger.info(f"Requesting matrix URL: {url}")
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
        url = f"{self.matrix_url}/matrix/driving-car"
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
                    
                    logger.info(f"Requesting batched matrix URL: {url}")
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

    def _get_directions_single_request(self, route_coordinates: List[Coords]) -> Dict[str, Any]:
        """
        Make a single directions API request for a route segment.
        Returns dict with 'geometry', 'distance', 'duration', or raises exception on failure.
        """
        # ORS expects [lon, lat]
        formatted_coords = [[coord[1], coord[0]] for coord in route_coordinates]

        url = f"{self.directions_url}/v2/directions/driving-car"
        headers = {
            'Authorization': self.api_key,
            'Content-Type': 'application/json'
        }
        body = {
            "coordinates": formatted_coords
        }

        logger.info(f"Requesting directions URL: {url} with {len(route_coordinates)} waypoints")
        response = self.session.post(
            url,
            json=body,
            headers=headers,
            timeout=self.timeout
        )

        # Enhanced error handling: log response.text on non-200 status
        if response.status_code != 200:
            logger.error(f"ORS API returned status {response.status_code}: {response.text}")
            response.raise_for_status()

        data = response.json()

        # Extract geometry from routes response
        geometry = []
        distance = 0.0
        duration = 0.0

        if 'routes' in data and len(data['routes']) > 0:
            route = data['routes'][0]

            # Extract geometry
            if 'geometry' in route:
                if isinstance(route['geometry'], str):
                    # Polyline string - decode it
                    geometry = decode_polyline(route['geometry']) or []
                elif isinstance(route['geometry'], dict) and 'coordinates' in route['geometry']:
                    # Coordinate array - convert [lon, lat] -> (lat, lon)
                    geometry = [(coord[1], coord[0]) for coord in route['geometry']['coordinates']]

            # Extract distance and duration from summary
            if 'summary' in route:
                summary = route['summary']
                distance = summary.get('distance', 0.0)  # meters
                duration = summary.get('duration', 0.0)  # seconds

        return {
            'geometry': geometry,
            'distance': distance,
            'duration': duration
        }

    @retry_with_backoff(max_attempts=3)
    def get_directions(self, route_coordinates: List[Coords]) -> Optional[Dict[str, Any]]:
        """
        Fetch route geometry (polylines) and precise distance/duration from ORS Directions API.
        Handles large routes by chunking them into smaller segments with overlapping waypoints.
        
        When a coordinate fails (unroutable), it is skipped and processing continues.
        Returns dict with 'geometry', 'distance', 'duration', and 'skipped_coordinates'.
        Returns None only on complete failure.
        """
        if not self.api_key:
            raise ValueError("API Key is required for directions requests.")

        if not route_coordinates or len(route_coordinates) < 2:
            raise ValueError("At least two coordinates are required for directions.")

        num_waypoints = len(route_coordinates)
        logger.info(f"Fetching directions for route with {num_waypoints} waypoints")
        
        # Track skipped coordinates (unroutable points)
        skipped_coordinates = []

        try:
            # Check if chunking is required
            if num_waypoints <= self.MAX_WAYPOINTS_PER_REQUEST:
                # Single request for small routes
                result = self._get_directions_with_skip(route_coordinates, skipped_coordinates)
                if result:
                    result['skipped_coordinates'] = skipped_coordinates
                return result
            else:
                # Chunking required for large routes
                logger.info(f"Route exceeds {self.MAX_WAYPOINTS_PER_REQUEST} waypoints, using chunking approach")

                # Create overlapping chunks with tracking of original indices
                chunks = []
                chunk_start_indices = []  # Track where each chunk starts in original coordinates
                for start_idx in range(0, num_waypoints - 1, self.MAX_WAYPOINTS_PER_REQUEST - 1):
                    end_idx = min(start_idx + self.MAX_WAYPOINTS_PER_REQUEST, num_waypoints)
                    chunk = route_coordinates[start_idx:end_idx]
                    chunks.append(chunk)
                    chunk_start_indices.append(start_idx)

                logger.info(f"Created {len(chunks)} chunks for route segmentation")

                # Process each chunk and aggregate results
                total_distance = 0.0
                total_duration = 0.0
                aggregated_geometry = []

                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)} with {len(chunk)} waypoints")

                    chunk_result = self._get_directions_with_skip(chunk, skipped_coordinates)
                    
                    if chunk_result:
                        # Aggregate metrics
                        total_distance += chunk_result['distance']
                        total_duration += chunk_result['duration']

                        # Aggregate geometry - skip the first point for chunks after the first to avoid duplicates
                        chunk_geometry = chunk_result['geometry']
                        if i > 0 and chunk_geometry:
                            # Skip the first point to avoid duplication at chunk boundaries
                            chunk_geometry = chunk_geometry[1:]
                        aggregated_geometry.extend(chunk_geometry)
                    else:
                        # Chunk completely failed even after skipping - use straight lines for this chunk
                        logger.warning(f"Chunk {i+1} failed completely, using straight-line fallback for this segment")
                        for coord in chunk:
                            aggregated_geometry.append((coord[0], coord[1]))

                return {
                    'geometry': aggregated_geometry,
                    'distance': total_distance,
                    'duration': total_duration,
                    'skipped_coordinates': skipped_coordinates
                }

        except requests.exceptions.Timeout as e:
            logger.error(f"Directions request timed out: {str(e)}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Directions request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in get_directions: {str(e)}")
            return None

    def _get_directions_with_skip(self, coordinates: List[Coords], 
                                   skipped_list: List[Coords]) -> Optional[Dict[str, Any]]:
        """
        Try to get directions for coordinates, skipping any unroutable points.
        Modifies skipped_list in place to track skipped coordinates.
        """
        current_coords = list(coordinates)
        max_skip_attempts = len(coordinates) - 2  # Need at least 2 points for a route
        
        for attempt in range(max_skip_attempts + 1):
            if len(current_coords) < 2:
                logger.warning("Not enough coordinates left after skipping, cannot get directions")
                return None
                
            try:
                return self._get_directions_single_request(current_coords)
            except Exception as e:
                error_str = str(e)
                
                # Try to parse which coordinate failed
                bad_idx = _parse_bad_coordinate_index(error_str)
                
                if bad_idx is not None and 0 <= bad_idx < len(current_coords):
                    bad_coord = current_coords[bad_idx]
                    logger.warning(f"Skipping unroutable coordinate at index {bad_idx}: {bad_coord}")
                    skipped_list.append(bad_coord)
                    
                    # Remove the bad coordinate and retry
                    current_coords = current_coords[:bad_idx] + current_coords[bad_idx + 1:]
                else:
                    # Cannot parse error or index out of range - give up on this chunk
                    logger.error(f"Cannot identify bad coordinate from error: {error_str}")
                    return None
        
        logger.warning("Max skip attempts reached")
        return None

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