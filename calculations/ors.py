"""
OpenRouteService API interactions with enhanced timeout and retry logic.
"""
import requests
import logging
import math
import time
from typing import List, Dict, Any, Tuple, Optional, Union
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .types import MatrixData, Coords, RouteMetrics
from .utils import retry_with_backoff, decode_polyline

logger = logging.getLogger(__name__)

# Default timeouts in seconds
DEFAULT_CONNECT_TIMEOUT = 10
DEFAULT_READ_TIMEOUT = 120  # Increased from 30 to 120 seconds
MAX_RETRIES = 3

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
        Fetch distance and duration matrix with enhanced error handling.
        
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

    @retry_with_backoff(max_attempts=3)
    def optimize_route(self, request_body: Dict[str, Any], timeout: Optional[Union[float, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Optimize vehicle routing with enhanced error handling.
        
        Args:
            request_body: The optimization request body
            timeout: Optional timeout in seconds (can be tuple of (connect_timeout, read_timeout))
            
        Returns:
            The optimization response as a dictionary
            
        Raises:
            ValueError: If API key is missing or request body is invalid
            requests.exceptions.RequestException: For request-related errors
        """
        if not self.api_key:
            raise ValueError("API Key is required for optimization requests.")
            
        if not request_body:
            raise ValueError("Request body cannot be empty.")
            
        logger.info("Starting route optimization")
        
        # Use instance timeout if not specified
        timeout = timeout or self.timeout
        
        try:
            url = "https://api.openrouteservice.org/optimization"
            headers = {
                'Authorization': self.api_key,
                'Content-Type': 'application/json'
            }
            
            response = self.session.post(
                url, 
                json=request_body, 
                headers=headers, 
                timeout=timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Optimization request timed out after {timeout} seconds: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Optimization request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {error_details}")
                except:
                    logger.error(f"Response content: {e.response.text}")
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