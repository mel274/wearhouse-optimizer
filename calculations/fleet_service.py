"""
Fleet management service for calculating truck capacities and fleet metrics.
Handles the business logic for fleet optimization calculations.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class FleetService:
    """Service for fleet-related calculations and metrics."""

    def calculate_capacities(self, total_volume: float, total_quantity: int,
                           fleet_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate truck capacities based on volume and quantity data.

        Args:
            total_volume: Total volume in cubic meters
            total_quantity: Total quantity of items
            fleet_settings: Dictionary with big_truck_vol, small_truck_vol, safety_factor

        Returns:
            Dictionary with small_capacity, big_capacity, and avg_tire_vol
        """
        logger.info(f"Calculating capacities for volume={total_volume}, quantity={total_quantity}")

        # Calculate average tire volume
        if total_quantity > 0:
            avg_tire_vol = total_volume / total_quantity
        else:
            avg_tire_vol = 0.00001  # Fallback to prevent division by zero

        big_truck_vol = fleet_settings['big_truck_vol']
        small_truck_vol = fleet_settings['small_truck_vol']
        safety_factor = fleet_settings['safety_factor']

        # Calculate separate capacities for big and small trucks
        big_capacity = int((big_truck_vol * safety_factor) / avg_tire_vol) if avg_tire_vol > 0.00001 else 0
        small_capacity = int((small_truck_vol * safety_factor) / avg_tire_vol) if avg_tire_vol > 0.00001 else 0

        logger.info(f"Calculated capacities: big={big_capacity}, small={small_capacity}, avg_vol={avg_tire_vol:.4f}")

        return {
            'small_capacity': small_capacity,
            'big_capacity': big_capacity,
            'avg_tire_vol': avg_tire_vol
        }
