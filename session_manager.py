"""
Session state management for Warehouse Location & Route Optimizer.
"""

import streamlit as st
import logging
from typing import Any, Optional
from exceptions import DataValidationError

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages Streamlit session state with validation and organization."""
    
    def __init__(self):
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize all session state variables with default values."""
        defaults = {
            'data': None,
            'solution': None,
            'geo_service': None,
            'warehouse_coords': None,
            'file_id': None,
            'optimization_params': {
                'fleet_size': None,
                'truck_capacity': None,
                'max_shift_hours': None,
                'service_time_minutes': None
            },
            'last_geocode_batch': None
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                logger.debug(f"Initialized session state: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get session state value with validation."""
        if key not in st.session_state:
            logger.warning(f"Session state key '{key}' not found, returning default")
            return default
        return st.session_state[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set session state value with logging."""
        try:
            st.session_state[key] = value
            logger.debug(f"Set session state: {key}")
        except Exception as e:
            logger.error(f"Failed to set session state '{key}': {e}")
            raise DataValidationError(f"Session state error: {e}")
    
    def clear_data(self) -> None:
        """Clear data-related session state."""
        data_keys = ['data', 'solution', 'warehouse_coords', 'file_id', 'last_geocode_batch']
        for key in data_keys:
            if key in st.session_state:
                st.session_state[key] = None
        logger.info("Cleared data-related session state")
    
    def validate_data_state(self) -> bool:
        """Validate that required data is present and valid."""
        data = self.get('data')
        if data is None or data.empty:
            return False
        
        # Check for required coordinates
        if 'lat' not in data.columns or 'lng' not in data.columns:
            return False
        
        # Check if any coordinates are present
        valid_coords = data.dropna(subset=['lat', 'lng'])
        return not valid_coords.empty
    
    def get_optimization_summary(self) -> dict:
        """Get summary of current optimization state."""
        summary = {
            'has_data': self.get('data') is not None,
            'has_solution': self.get('solution') is not None,
            'has_geocoded_data': self.validate_data_state(),
            'file_loaded': self.get('file_id') is not None,
            'geo_service_active': self.get('geo_service') is not None
        }
        
        if summary['has_data']:
            data = self.get('data')
            summary['customer_count'] = len(data)
            summary['geocoded_count'] = data[['lat', 'lng']].notna().all(axis=1).sum()
        
        if summary['has_solution']:
            solution = self.get('solution')
            summary['vehicles_used'] = solution.get('metrics', {}).get('num_vehicles_used', 0)
            summary['total_distance'] = solution.get('metrics', {}).get('total_distance', 0)
        
        return summary
