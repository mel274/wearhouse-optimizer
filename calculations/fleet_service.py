"""
Fleet management service for calculating truck capacities and fleet metrics.
Handles the business logic for fleet optimization calculations.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class FleetService:
    """Service for fleet-related calculations and metrics."""

    def calculate_capacities(self, fleet_settings: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract truck capacity volumes from fleet settings.

        Args:
            fleet_settings: Dictionary with big_truck_vol and small_truck_vol

        Returns:
            Dictionary with small_capacity and big_capacity (in cubic meters)
        """
        big_truck_vol = fleet_settings['big_truck_vol']
        small_truck_vol = fleet_settings['small_truck_vol']

        logger.info(f"Extracted capacities: big={big_truck_vol}, small={small_truck_vol}")

        return {
            'small_capacity': small_truck_vol,
            'big_capacity': big_truck_vol
        }
