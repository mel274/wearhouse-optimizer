"""
Configuration constants and settings for Warehouse Location & Route Optimizer.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration constants."""
    
    # OpenRouteService API Key (Primary)
    OPENROUTESERVICE_API_KEY = os.getenv('OPENROUTESERVICE_API_KEY', '')
    
    # Radar API Key (Backup)
    RADAR_API_KEY = os.getenv('RADAR_API_KEY', '')
    
    # Fleet and operation parameters
    DEFAULT_FLEET_SIZE = 17
    DEFAULT_CAPACITY = 400
    MAX_SHIFT_HOURS = 12
    SERVICE_TIME_MINUTES = 15
    
    # Optimization algorithm parameters
    MAX_ITERATIONS = 50
    CANDIDATE_PERCENTAGE = 0.3
    MIN_K_START = 1
    
    # Geographic and visualization parameters
    ISRAEL_LAT_MIN = 29.0
    ISRAEL_LAT_MAX = 34.0
    ISRAEL_LNG_MIN = 34.0
    ISRAEL_LNG_MAX = 36.0
    ROUTE_INFLUENCE_RADIUS = 2000  # meters
    DEFAULT_CENTER_LAT = 31.7683  # Jerusalem
    DEFAULT_CENTER_LNG = 35.2137

    # Geographic restriction zones
    GUSH_DAN_BOUNDS = {
        'min_lat': 32.02,
        'max_lat': 32.15,
        'min_lon': 34.74,
        'max_lon': 34.85
    }

    # Fleet defaults for volume-based calculations
    FLEET_DEFAULTS = {
        'big_truck_vol': 32.0,
        'small_truck_vol': 27.0,
        'safety_factor': 0.7
    }
    
    # API and caching parameters
    API_TIMEOUT = 30 # Increased timeout for robustness
    API_RETRY_ATTEMPTS = 3
    CACHE_MAX_SIZE = 10000
    CACHE_CLEANUP_THRESHOLD = 0.8
    
    # --- GRACEFUL EXECUTION SETTINGS ---
    RATE_LIMIT_DELAY = 2.0  # Wait 2 seconds between API calls (Prevents overloading)
    STEP_DELAY = 3.0        # Wait 3 seconds between major logic stages
    
    # Expected Hebrew column names for Excel uploads
    EXPECTED_HEBREW_COLUMNS = [
        'כתובת',
        'מס\' לקוח',
        'שם לקוח',
        'תאריך אספקה',
        'כמות',
        'שם קו'
    ]