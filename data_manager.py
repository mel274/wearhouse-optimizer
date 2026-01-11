"""
Data management service for Excel file processing and customer aggregation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from config import Config
import shared.utils as utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """Service for loading, validating, and aggregating delivery data."""
    
    def __init__(self):
        """Initialize DataManager with default configuration."""
        self.expected_columns = Config.EXPECTED_HEBREW_COLUMNS
    
    def load_data(self, file) -> pd.DataFrame:
        """Load and validate Excel/CSV file with delivery data."""
        try:
            # Use utils function for robust file handling (Excel and CSV)
            df = utils.validate_hebrew_encoding(file)
            
            if df.empty:
                raise ValueError("File is empty or cannot be read")
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace(r'[\\"]', '', regex=True).str.replace('\xa0', ' ')
            
            # Validate required columns
            missing_columns = []
            found_columns = list(df.columns)
            
            for col in self.expected_columns:
                if col not in df.columns:
                    missing_columns.append(col)
            
            if missing_columns:
                error_msg = f"Missing columns in file:\n\n"
                error_msg += f"Expected columns: {', '.join(self.expected_columns)}\n"
                error_msg += f"Found columns: {', '.join(found_columns)}\n"
                error_msg += f"Missing columns: {', '.join(missing_columns)}"
                raise ValueError(error_msg)
            
            # Clean data
            df = self._clean_data(df)
            
            logger.info(f"Loaded {len(df)} rows from file")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise ValueError(f"Error loading file: {str(e)}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess raw delivery data."""
        df = df.copy()
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['כתובת', 'מס\' לקוח', 'שם לקוח'])
        
        # Convert date column to datetime
        if 'תאריך אספקה' in df.columns:
            df['תאריך אספקה'] = pd.to_datetime(df['תאריך אספקה'], errors='coerce')
        
        # Convert quantity to numeric
        if 'כמות' in df.columns:
            df['כמות'] = pd.to_numeric(df['כמות'], errors='coerce').fillna(0)

        # --- NEW: Volume Calculation ---
        volume_col_hebrew = 'נפח פריט (מטר מעוקב)'
        volume_col_english = 'volume_cbm'

        volume_col = None
        if volume_col_hebrew in df.columns:
            volume_col = volume_col_hebrew
        elif volume_col_english in df.columns:
            volume_col = volume_col_english

        if volume_col:
            logger.info(f"Volume column '{volume_col}' found. Calculating total volume.")
            df[volume_col] = pd.to_numeric(df[volume_col], errors='coerce').fillna(0)
            # Volume is already in Cubic Meters (m³), so multiply unit volume by quantity
            df['total_volume_m3'] = df[volume_col] * df['כמות']
        else:
            logger.warning(f"No volume column found ('{volume_col_hebrew}' or '{volume_col_english}'). Defaulting total volume to 0.")
            df['total_volume_m3'] = 0
        
        # Strip whitespace from string columns
        string_columns = ['כתובת', 'מס\' לקוח', 'שם לקוח', 'שם קו']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
    
    def aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate raw delivery data by customer (one row per customer).
        Calculates Customer Force (80th percentile) for route planning.
        """
        try:
            # Group by customer
            group_cols = ['מס\' לקוח', 'שם לקוח', 'כתובת']
            
            missing_group_cols = [col for col in group_cols if col not in df.columns]
            if missing_group_cols:
                raise ValueError(f"Missing group columns: {missing_group_cols}")
            
            # Step 1: Daily Aggregation - Group by Customer + Date to get daily order sizes
            daily_group_cols = ['מס\' לקוח', 'תאריך אספקה']
            daily_agg_ops = {
                'כמות': 'sum'
            }
            if 'total_volume_m3' in df.columns:
                daily_agg_ops['total_volume_m3'] = 'sum'
            
            daily_orders = df.groupby(daily_group_cols, as_index=False).agg(daily_agg_ops)
            logger.info(f"Created daily order aggregation: {len(daily_orders)} customer-date pairs")
            
            # Step 2: Calculate Force (Percentile) - Group by Customer and calculate percentile
            percentile_value = Config.CUSTOMER_FORCE_PERCENTILE * 100
            
            def calculate_percentile(sorted_array, percentile):
                """Calculate percentile, handling edge cases."""
                if len(sorted_array) == 0:
                    return 0.0
                if len(sorted_array) == 1:
                    return float(sorted_array[0])
                return float(np.percentile(sorted_array, percentile, method='linear'))
            
            # Use apply() instead of agg() to ensure proper calculation
            def calc_force_metrics(group):
                """Calculate force metrics for a customer group."""
                # Sort the daily values in ascending order (growing up series)
                sorted_quantities = np.sort(group['כמות'].values)
                
                result = {
                    'force_quantity': calculate_percentile(sorted_quantities, percentile_value)
                }
                
                if 'total_volume_m3' in group.columns:
                    sorted_volumes = np.sort(group['total_volume_m3'].values)
                    result['force_volume'] = calculate_percentile(sorted_volumes, percentile_value)
                else:
                    result['force_volume'] = 0.0
                
                return pd.Series(result)
            
            force_metrics = daily_orders.groupby('מס\' לקוח').apply(calc_force_metrics, include_groups=False).reset_index()
            
            logger.info(f"Calculated force metrics for {len(force_metrics)} customers using {percentile_value}th percentile")
            
            # Step 3: Main aggregation (for other metrics)
            agg_ops = {
                'תאריך אספקה': 'nunique',
                'כמות': 'sum',
                'שם קו': lambda x: x.mode().iloc[0] if not x.mode().empty else ''
            }
            if 'total_volume_m3' in df.columns:
                agg_ops['total_volume_m3'] = 'sum'

            # Perform main aggregation
            aggregated = df.groupby(group_cols, as_index=False).agg(agg_ops)
            
            # Rename columns
            aggregated = aggregated.rename(columns={
                'תאריך אספקה': 'num_visits',
                'כמות': 'total_quantity',
                'שם קו': 'current_route'
            })
            
            # Step 4: Merge Force metrics back into aggregated dataframe
            merge_cols = ['מס\' לקוח', 'force_quantity', 'force_volume']
            aggregated = aggregated.merge(
                force_metrics[merge_cols],
                on='מס\' לקוח',
                how='left'
            )
            
            # Fill missing force values (shouldn't happen, but defensive)
            aggregated['force_quantity'] = aggregated['force_quantity'].fillna(0)
            aggregated['force_volume'] = aggregated['force_volume'].fillna(0)
            
            # Use force_quantity as avg_quantity for seamless integration with existing code
            aggregated['avg_quantity'] = aggregated['force_quantity']
            
            # Also keep the traditional average for reference
            aggregated['avg_quantity_traditional'] = aggregated['total_quantity'] / aggregated['num_visits']
            
            # Add coordinate columns (will be populated later)
            aggregated['lat'] = None
            aggregated['lng'] = None
            
            logger.info(f"Aggregated {len(df)} items to {len(aggregated)} customers using {percentile_value}th percentile force")
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            raise ValueError(f"שגיאה באגרגציה של הנתונים: {str(e)}")
    
    def recalculate_customer_force(self, aggregated_df: pd.DataFrame, raw_df: pd.DataFrame, percentile: float) -> pd.DataFrame:
        """
        Recalculate Customer Force metrics based on a percentile value.
        
        Args:
            aggregated_df: The aggregated customer dataframe (with existing force metrics)
            raw_df: The raw delivery data (before aggregation)
            percentile: The percentile value (0.0 to 1.0) to use for force calculation
            
        Returns:
            Updated aggregated dataframe with new force metrics
        """
        try:
            # Step 1: Daily Aggregation - Group by Customer ID and Date to get daily order sizes
            daily_group_cols = ['מס\' לקוח', 'תאריך אספקה']
            daily_agg_ops = {
                'כמות': 'sum'
            }
            if 'total_volume_m3' in raw_df.columns:
                daily_agg_ops['total_volume_m3'] = 'sum'
            
            daily_orders = raw_df.groupby(daily_group_cols, as_index=False).agg(daily_agg_ops)
            logger.info(f"Recalculating force: {len(daily_orders)} customer-date pairs")
            
            # Step 2: Calculate New Percentile - Group by Customer and calculate percentile
            percentile_value = percentile * 100
            
            def calculate_percentile(sorted_array, pct):
                """Calculate percentile, handling edge cases."""
                if len(sorted_array) == 0:
                    return 0.0
                if len(sorted_array) == 1:
                    return float(sorted_array[0])
                return float(np.percentile(sorted_array, pct, method='linear'))
            
            # Use apply() instead of agg() to ensure proper calculation
            def calc_force_metrics(group):
                """Calculate force metrics for a customer group."""
                # Sort the daily values in ascending order (growing up series)
                sorted_quantities = np.sort(group['כמות'].values)
                
                result = {
                    'force_quantity': calculate_percentile(sorted_quantities, percentile_value)
                }
                
                if 'total_volume_m3' in group.columns:
                    sorted_volumes = np.sort(group['total_volume_m3'].values)
                    result['force_volume'] = calculate_percentile(sorted_volumes, percentile_value)
                else:
                    result['force_volume'] = 0.0
                
                return pd.Series(result)
            
            force_metrics = daily_orders.groupby('מס\' לקוח').apply(calc_force_metrics, include_groups=False).reset_index()
            
            logger.info(f"Recalculated force metrics for {len(force_metrics)} customers using {percentile_value}th percentile")
            
            # Step 3: Merge new force metrics into aggregated dataframe
            # Create a copy to avoid modifying the original
            updated_df = aggregated_df.copy()
            
            # Drop existing force columns if they exist (we'll replace them)
            if 'force_quantity' in updated_df.columns:
                updated_df = updated_df.drop(columns=['force_quantity'])
            if 'force_volume' in updated_df.columns:
                updated_df = updated_df.drop(columns=['force_volume'])
            
            # Merge on customer ID
            merge_cols = ['מס\' לקוח', 'force_quantity', 'force_volume']
            updated_df = updated_df.merge(
                force_metrics[merge_cols],
                on='מס\' לקוח',
                how='left'
            )
            
            # Fill missing force values (shouldn't happen, but defensive)
            updated_df['force_quantity'] = updated_df['force_quantity'].fillna(0)
            updated_df['force_volume'] = updated_df['force_volume'].fillna(0)
            
            # Overwrite avg_quantity with the new force_quantity
            updated_df['avg_quantity'] = updated_df['force_quantity']
            
            logger.info(f"Updated customer force metrics with {percentile_value}th percentile")
            return updated_df
            
        except Exception as e:
            logger.error(f"Error recalculating customer force: {e}")
            raise ValueError(f"שגיאה בחישוב מחדש של Customer Force: {str(e)}")
    
    def calculate_center_of_gravity(self, df: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Calculate weighted center of gravity."""
        try:
            valid_coords = df.dropna(subset=['lat', 'lng'])
            
            if valid_coords.empty:
                return {'lat': None, 'lng': None}
            
            if 'num_visits' not in valid_coords.columns:
                weights = np.ones(len(valid_coords))
            else:
                weights = valid_coords['num_visits'].fillna(1)
            
            total_weight = weights.sum()
            if total_weight == 0:
                return {'lat': None, 'lng': None}
            
            weighted_lat = (valid_coords['lat'] * weights).sum() / total_weight
            weighted_lng = (valid_coords['lng'] * weights).sum() / total_weight
            
            return {
                'lat': float(weighted_lat),
                'lng': float(weighted_lng)
            }
            
        except Exception as e:
            logger.error(f"Error calculating center of gravity: {e}")
            return {'lat': None, 'lng': None}
    
    def add_coordinates(self, df: pd.DataFrame, geo_service) -> pd.DataFrame:
        """Add coordinates to customer data using geocoding service."""
        df_copy = df.copy()
        
        for idx, row in df_copy.iterrows():
            if pd.isna(row.get('lat')) or pd.isna(row.get('lng')):
                address = row['כתובת']
                lat, lng = geo_service.get_coordinates(address)
                df_copy.at[idx, 'lat'] = lat
                df_copy.at[idx, 'lng'] = lng
        
        return df_copy