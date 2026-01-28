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

    # Fleet defaults for volume-based calculations
    FLEET_DEFAULTS = {
        'big_truck_vol': 32.0,
        'small_truck_vol': 27.0,
        'safety_factor': 0.7
    }

    # Exception tolerance for simulation (retry mechanism)
    # Percentage of total drives that can exceed limits before retry
    AVAILABLE_EXCEPTION_PERCENT = 0.03  # 3% of total drives can exceed limits

    # Master Route Time Multiplier for dynamic time horizon
    # Master routes use soft constraints: horizon = UI time × multiplier
    # This allows flexibility for territory planning (master routes include ALL customers,
    # but daily routes only serve a subset)
    MASTER_ROUTE_TIME_MULTIPLIER = 1  # 1.5× UI time as hard cap for master routes

    # --- OPTIMIZATION COSTS ---
    # Fleet Squeeze - Nuclear Pricing:
    # Stage 1 (Highest): Serve ALL customers (MANDATORY - no disjunctions)
    # Stage 2 (Middle): Pack vehicles tightly with AUTOMATIC strategy
    # Stage 3 (Lowest): Minimize fleet size with NUCLEAR cost (Cost: 100,000,000 per truck)
    UNSERVED_PENALTY = 100_000_000_000  # Penalty for not serving a customer (100B)
    VEHICLE_FIXED_COST = 100_000_000  # Fixed cost per vehicle used (100M) - nuclear pricing
    
    # Customer Force (Percentile-based demand planning)
    # Deprecated: percentile-based Customer Force (kept for backward compatibility).
    CUSTOMER_FORCE_PERCENTILE = 0.8
    
    # API and caching parameters
    API_TIMEOUT = 30 # Increased timeout for robustness
    API_RETRY_ATTEMPTS = 3
    CACHE_MAX_SIZE = 10000
    CACHE_CLEANUP_THRESHOLD = 0.8

    # OpenRouteService specific parameters
    ORS_CONNECT_TIMEOUT = 10
    ORS_READ_TIMEOUT = 120  # Increased from 30 to 120 seconds
    ORS_MAX_RETRIES = 3
    ORS_MATRIX_CHUNK_SIZE = 2000  # only relevant for large location sets when using api not loacl docker.
    
  
    
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