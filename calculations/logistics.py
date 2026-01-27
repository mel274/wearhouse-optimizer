"""
Logistics calculations for fleet sizing and geographic restrictions.
"""

import pandas as pd
import math
from config import Config
from typing import Dict, Any
from shared.geo_utils import is_in_small_truck_zone

def calculate_fleet_metrics(df: pd.DataFrame, fleet_settings: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculates the required number of trucks based on volume and geographic zone.

    Args:
        df: Aggregated customer data with 'lat', 'lng', and 'total_volume_m3'.
        fleet_settings: Dictionary with truck volumes and safety factor.

    Returns:
        DataFrame with new columns: 'truck_type', 'trucks_needed', 'color'.
    """
    df_copy = df.copy()

    big_truck_vol = fleet_settings.get('big_truck_vol', Config.FLEET_DEFAULTS['big_truck_vol'])
    small_truck_vol = fleet_settings.get('small_truck_vol', Config.FLEET_DEFAULTS['small_truck_vol'])
    safety_factor = fleet_settings.get('safety_factor', Config.FLEET_DEFAULTS['safety_factor'])

    truck_types = []
    trucks_needed = []
    colors = []

    for _, row in df_copy.iterrows():
        lat, lon = row.get('lat'), row.get('lng')
        volume = row.get('total_volume_m3', 0)

        if is_in_small_truck_zone(lat, lon):
            truck_type = "Small Truck"
            truck_vol = small_truck_vol
            color = "#FF0000"  # Red
        else:
            truck_type = "Big Truck"
            truck_vol = big_truck_vol
            color = "#0000FF"  # Blue

        effective_capacity = truck_vol * safety_factor
        
        needed = 0
        if effective_capacity > 0:
            needed = math.ceil(volume / effective_capacity)
        
        truck_types.append(truck_type)
        trucks_needed.append(needed)
        colors.append(color)

    df_copy['truck_type'] = truck_types
    df_copy['trucks_needed'] = trucks_needed
    df_copy['color'] = colors

    return df_copy
