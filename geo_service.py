"""
Geocoding service using OpenRouteService (ORS) as primary.
Cleaned up to remove optimization logic (now in calculations.ors).
"""

import requests
import pickle
import os
import time
import logging
from typing import List, Tuple, Dict, Optional, Any
from config import Config
from exceptions import GeocodingError, CacheError
from shared.utils import validate_address, validate_coordinates, decode_polyline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeoService:
    """Service for Geocoding operations."""
    
    CACHE_FILE = "geocode_cache.pkl"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.geocode_cache = self._load_cache()
        self.last_api_call = 0
        
        if not self.api_key:
            logging.warning("GeoService initialized without API key")
    
    def _load_cache(self) -> Dict[str, Tuple[float, float]]:
        """Load geocoding cache with size management."""
        if os.path.exists(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, 'rb') as f:
                    cache = pickle.load(f)
                    if len(cache) > Config.CACHE_MAX_SIZE:
                        cache = dict(list(cache.items())[-Config.CACHE_MAX_SIZE:])
                    logger.info(f"Loaded {len(cache)} cached geocoding entries")
                    return cache
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self) -> None:
        """Save cache with cleanup if needed."""
        try:
            if len(self.geocode_cache) > Config.CACHE_MAX_SIZE * Config.CACHE_CLEANUP_THRESHOLD:
                self._cleanup_cache()
            
            with open(self.CACHE_FILE, 'wb') as f:
                pickle.dump(self.geocode_cache, f)
            logger.debug(f"Saved {len(self.geocode_cache)} cached geocoding entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            raise CacheError(f"Cache save failed: {e}")
    
    def _cleanup_cache(self):
        """Remove oldest entries from cache to maintain size limits."""
        entries_to_remove = len(self.geocode_cache) - Config.CACHE_MAX_SIZE
        if entries_to_remove > 0:
            keys_to_remove = list(self.geocode_cache.keys())[:entries_to_remove]
            for key in keys_to_remove:
                del self.geocode_cache[key]
    
    def _rate_limit(self):
        """Implement rate limiting between API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_api_call
        if time_since_last < Config.RATE_LIMIT_DELAY:
            sleep_time = Config.RATE_LIMIT_DELAY - time_since_last
            time.sleep(sleep_time)
        self.last_api_call = time.time()
    
    def snap_to_road(self, coords: Tuple[float, float]) -> Tuple[float, float]:
        """Snaps a coordinate to the nearest road using ORS reverse geocoding."""
        if not self.api_key:
            return coords

        url = "http://localhost:8080/ors/geocode/reverse"
        params = {
            'api_key': self.api_key,
            'point.lon': coords[1],
            'point.lat': coords[0],
            'boundary.country': 'IL',
            'size': 1
        }
        try:
            self._rate_limit()
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('features'):
                    # ORS returns [lon, lat]
                    snapped_lon, snapped_lat = data['features'][0]['geometry']['coordinates']
                    snapped_coords = (snapped_lat, snapped_lon)
                    return snapped_coords
        except Exception as e:
            logger.warning(f"Could not snap coordinate {coords} to road: {e}")
        
        return coords

    def get_route_polyline(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Get route polyline between two points using ORS Directions API.
        Used for visualization fallback.
        """
        if not self.api_key:
            return [start, end]

        url = "https://api.openrouteservice.org/v2/directions/driving-car"

        # ORS expects [lon, lat]
        coordinates = [[start[1], start[0]], [end[1], end[0]]]
        body = {"coordinates": coordinates}
        headers = {'Authorization': self.api_key, 'Content-Type': 'application/json'}
        
        try:
            self._rate_limit()
            response = requests.post(url, json=body, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'routes' in data and len(data['routes']) > 0:
                    route = data['routes'][0]
                    if 'geometry' in route:
                        if isinstance(route['geometry'], str):
                            decoded = decode_polyline(route['geometry'])
                            if decoded: return decoded
                        elif 'coordinates' in route['geometry']:
                            return [(lat, lng) for lng, lat in route['geometry']['coordinates']]
        except Exception:
            pass
        return [start, end]

    def get_coordinates(self, address: str) -> Tuple[Optional[float], Optional[float]]:
        """Get coordinates for an address with validation, caching, and multiple services."""
        try:
            sanitized_address = validate_address(address)
            
            if sanitized_address in self.geocode_cache:
                return self.geocode_cache[sanitized_address]
            
            # Priority: Nominatim -> ORS -> GovMap -> Photon -> City
            services_order = [
                ("Nominatim", self._geocode_nominatim),
                ("OpenRouteService", self._geocode_ors),
                ("GovMap Israel", self._geocode_govmap),
                ("Photon", self._geocode_photon),
                ("City-Only", self._geocode_city_only)
            ]

            for _, geocoding_func in services_order:
                try:
                    self._rate_limit()
                    lat, lng = geocoding_func(sanitized_address)
                    if lat and lng and validate_coordinates(lat, lng):
                        self.geocode_cache[sanitized_address] = (lat, lng)
                        self._save_cache()
                        return lat, lng
                except Exception:
                    continue
            
            error_msg = f"All geocoding services failed for address: '{sanitized_address}'"
            logger.error(error_msg)
            raise GeocodingError(error_msg)
            
        except Exception:
            return None, None

    # --- Geocoding Service Implementation Methods ---
    def _geocode_nominatim(self, address: str) -> Tuple[Optional[float], Optional[float]]:
        url = "https://nominatim.openstreetmap.org/search"
        headers = {'User-Agent': 'WarehouseOptimizer/1.0', 'Referer': 'http://localhost'}
        params = {'q': address, 'format': 'json', 'limit': 1, 'countrycodes': 'il'}
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data: return float(data[0]['lat']), float(data[0]['lon'])
        return None, None

    def _geocode_ors(self, address: str) -> Tuple[Optional[float], Optional[float]]:
        if not self.api_key: return None, None
        url = "http://localhost:8080/ors/geocode/search"
        params = {'api_key': self.api_key, 'text': address, 'boundary.country': 'IL', 'size': 1}
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'features' in data and len(data['features']) > 0:
                coords = data['features'][0]['geometry']['coordinates']
                return float(coords[1]), float(coords[0])
        return None, None

    def _geocode_govmap(self, address: str) -> Tuple[Optional[float], Optional[float]]:
        url = "https://es.govmap.gov.il/Taldor/GovMap/Svcs/Geocode.svc/json/Geocode"
        headers = {'Content-Type': 'application/json', 'Referer': 'https://www.govmap.gov.il/'}
        payload = {'keyword': address, 'type': 0, 'lang': 0}
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'Result' in data and data['Result']:
                    res = data['Result'][0]
                    if 'Y' in res and 'X' in res:
                        if 29.0 <= res['Y'] <= 33.5 and 34.0 <= res['X'] <= 36.0:
                            return float(res['Y']), float(res['X'])
        except Exception: pass
        return None, None

    def _geocode_photon(self, address: str) -> Tuple[Optional[float], Optional[float]]:
        url = "https://photon.komoot.io/api/"
        params = {'q': address, 'limit': 1}
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'features' in data and len(data['features']) > 0:
                coords = data['features'][0]['geometry']['coordinates']
                return float(coords[1]), float(coords[0])
        return None, None

    def _geocode_city_only(self, address: str) -> Tuple[Optional[float], Optional[float]]:
        try:
            parts = address.replace(',', ' ').split()
            if not parts: return None, None
            city_candidate = parts[0]
            if len(city_candidate) > 2: return self._geocode_nominatim(city_candidate)
        except Exception: pass
        return None, None