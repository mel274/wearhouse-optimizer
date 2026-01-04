"""
Data management service for Excel file processing and customer aggregation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from config import Config
import utils

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
            # Convert Liters to Cubic Meters (1 L = 0.001 m^3)
            df['total_volume_m3'] = (df[volume_col] * df['כמות']) / 1000
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
        Calculates average quantity per visit.
        """
        try:
            # Group by customer
            group_cols = ['מס\' לקוח', 'שם לקוח', 'כתובת']
            
            missing_group_cols = [col for col in group_cols if col not in df.columns]
            if missing_group_cols:
                raise ValueError(f"Missing group columns: {missing_group_cols}")
            
            # Define aggregation operations
            agg_ops = {
                'תאריך אספקה': 'nunique',
                'כמות': 'sum',
                'שם קו': lambda x: x.mode().iloc[0] if not x.mode().empty else ''
            }
            if 'total_volume_m3' in df.columns:
                agg_ops['total_volume_m3'] = 'sum'

            # Perform aggregation
            aggregated = df.groupby(group_cols, as_index=False).agg(agg_ops)
            
            # Rename columns
            aggregated = aggregated.rename(columns={
                'תאריך אספקה': 'num_visits',
                'כמות': 'total_quantity',
                'שם קו': 'current_route'
            })
            
            # --- NEW: Calculate Average Quantity per Visit ---
            # This is what we will use for daily route planning
            aggregated['avg_quantity'] = aggregated['total_quantity'] / aggregated['num_visits']
            
            # Add coordinate columns (will be populated later)
            aggregated['lat'] = None
            aggregated['lng'] = None
            
            logger.info(f"Aggregated {len(df)} items to {len(aggregated)} customers")
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            raise ValueError(f"שגיאה באגרגציה של הנתונים: {str(e)}")
    
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