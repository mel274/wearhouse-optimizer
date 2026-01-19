"""
Configuration constants and settings for Warehouse Location & Route Optimizer.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration constants."""
    # NEW: Point to your local server
    ORS_BASE_URL = "http://localhost:8080/ors/v2"
    # OpenRouteService API Key (Primary)
    OPENROUTESERVICE_API_KEY = os.getenv('OPENROUTESERVICE_API_KEY', '')
    
    # Radar API Key (Backup)
    RADAR_API_KEY = os.getenv('RADAR_API_KEY', '')
    
    # Fleet and operation parameters
    DEFAULT_FLEET_SIZE = 18
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
        'min_lat': 32.00,
        'max_lat': 32.12,
        'min_lon': 34.74,
        'max_lon': 34.90,
    }

    # Fleet defaults for volume-based calculations
    FLEET_DEFAULTS = {
        'big_truck_vol': 32.0,
        'small_truck_vol': 27.0,
        'safety_factor': 0.7
    }

    # --- OPTIMIZATION COSTS ---
    # Strict priority hierarchy enforcement:
    # Stage 1 (Highest): Serve ALL customers (Penalty: 10,000,000)
    # Stage 2 (Middle): Minimize the number of trucks (Cost: 1,000,000 per truck)
    # Stage 3 (Lowest): Minimize total kilometers (Cost: 1 per meter)
    UNSERVED_PENALTY = 10000000  # Penalty for not serving a customer
    VEHICLE_FIXED_COST = 1000000  # Fixed cost per vehicle used
    
    # Customer Force (Percentile-based demand planning)
    CUSTOMER_FORCE_PERCENTILE = 0.8  # 80th percentile for route planning
    
    # API and caching parameters
    API_TIMEOUT = 30 # Increased timeout for robustness
    API_RETRY_ATTEMPTS = 3
    CACHE_MAX_SIZE = 10000
    CACHE_CLEANUP_THRESHOLD = 0.8

    # OpenRouteService specific parameters
    ORS_CONNECT_TIMEOUT = 10
    ORS_READ_TIMEOUT = 120  # Increased from 30 to 120 seconds
    ORS_MAX_RETRIES = 3
    ORS_MATRIX_CHUNK_SIZE = 40  # ORS API limit is ~50 locations per request
    
    # --- GRACEFUL EXECUTION SETTINGS ---
    RATE_LIMIT_DELAY = 0.1  # Wait 0.1 seconds between API calls (Prevents overloading)
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