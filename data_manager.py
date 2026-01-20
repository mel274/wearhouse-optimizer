"""
Data management service for Excel file processing and customer aggregation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import os
from config import Config
import shared.utils as utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """Service for loading, validating, and aggregating delivery data."""

    def __init__(self):
        """Initialize DataManager with default configuration."""
        self.expected_columns = Config.EXPECTED_HEBREW_COLUMNS

    def _calculate_client_force(self, daily_values, total_days: int, buffer: float) -> Tuple[float, float]:
        """
        Calculate Client Force using industry standard formula:
            force = mean + (std * buffer)

        Critically, includes "days without orders" as 0 values, based on the
        total_days timeline of the uploaded file.

        Args:
            daily_values: Array-like of observed daily totals (e.g., quantities/volumes) for days with orders.
            total_days: Total number of days in the file's date range (inclusive).
            buffer: Multiplier applied to standard deviation.

        Returns:
            Tuple of (force_value, std_deviation) as floats.
        """
        try:
            total_days = int(total_days)
        except Exception:
            total_days = 1

        if total_days <= 0:
            total_days = 1

        values = np.asarray(daily_values, dtype=float)
        if values.ndim != 1:
            values = values.reshape(-1)

        num_zeros = max(0, total_days - len(values))
        if num_zeros > 0:
            full_values = np.concatenate([values, np.zeros(num_zeros, dtype=float)])
        else:
            full_values = values

        if full_values.size == 0:
            return 0.0

        mean = float(np.mean(full_values))
        std = float(np.std(full_values, ddof=1))
        force = mean + (std * float(buffer))
        return force, std

    def _normalize_name(self, name: str) -> str:
        """
        Normalize customer names by removing punctuation and fixing spacing.

        Args:
            name: Raw customer name string

        Returns:
            Normalized name with punctuation removed and spacing cleaned
        """
        import re

        # Convert to string and strip whitespace
        name = str(name).strip()

        # Remove punctuation: . , ' " ( ) / \
        # Keep hyphens (-) as they are often meaningful in company names
        punctuation = r'[.,\'\"\(\)/\\]'
        name = re.sub(punctuation, '', name)

        # Replace multiple consecutive spaces with single space
        name = re.sub(r'\s+', ' ', name)

        return name.strip()

    def _load_master_locations(self) -> Dict[str, Tuple[float, float]]:
        """
        Load master locations from Excel file for hybrid geocoding.

        Returns:
            Dictionary mapping customer names to (lat, lng) tuples.
            Returns empty dict if file doesn't exist or can't be loaded.
        """
        master_file_path = "assets/garages.xlsx"
        master_locations = {}

        # Check if file exists
        if not os.path.exists(master_file_path):
            logger.info(f"Master locations file not found: {master_file_path}. Using API geocoding only.")
            return master_locations

        try:
            # Load the Excel file
            master_df = pd.read_excel(master_file_path)

            # Check if required columns exist
            name_col = 'שם לקוח \\ מחסן'
            coords_col = 'קואורדינטות'

            if name_col not in master_df.columns or coords_col not in master_df.columns:
                logger.warning(f"Required columns '{name_col}' or '{coords_col}' not found in master file. Using API geocoding only.")
                return master_locations

            # Process each row
            for idx, row in master_df.iterrows():
                try:
                    # Get and normalize customer name
                    raw_customer_name = str(row.get(name_col, '')).strip()
                    if not raw_customer_name:
                        continue

                    normalized_name = self._normalize_name(raw_customer_name)

                    # Parse coordinates
                    coords_str = str(row.get(coords_col, '')).strip()
                    if not coords_str:
                        continue

                    # Split by comma and convert to float
                    coords_parts = coords_str.split(',')
                    if len(coords_parts) != 2:
                        logger.warning(f"Invalid coordinate format for customer '{raw_customer_name}' (normalized: '{normalized_name}'): {coords_str}")
                        continue

                    lat = float(coords_parts[0].strip())
                    lng = float(coords_parts[1].strip())

                    # Store in dictionary using normalized name as key
                    master_locations[normalized_name] = (lat, lng)

                except Exception as e:
                    logger.warning(f"Error parsing row {idx} in master locations file: {e}")
                    continue

            logger.info(f"Loaded {len(master_locations)} master locations from {master_file_path}")
            return master_locations

        except Exception as e:
            logger.error(f"Error loading master locations file: {e}")
            return master_locations

    def _update_api_detected_file(self, new_api_hits: Dict[str, Tuple[float, float]], master_locations: Dict[str, Tuple[float, float]]):
        """
        Update the API-detected locations file with new API hits and clean up customers
        that have been moved to the master file.

        Args:
            new_api_hits: Dictionary of customer names -> (lat, lng) for customers found via API in current run
            master_locations: Dictionary of customer names -> (lat, lng) from master file
        """
        detected_file_path = "assets/api_detected_locations.xlsx"

        # Load existing detected locations
        existing_detected = {}
        if os.path.exists(detected_file_path):
            try:
                detected_df = pd.read_excel(detected_file_path)
                name_col = 'שם לקוח \\ מחסן'
                coords_col = 'קואורדינטות'

                if name_col in detected_df.columns and coords_col in detected_df.columns:
                    for idx, row in detected_df.iterrows():
                        try:
                            customer_name = str(row.get(name_col, '')).strip()
                            coords_str = str(row.get(coords_col, '')).strip()

                            if customer_name and coords_str:
                                coords_parts = coords_str.split(',')
                                if len(coords_parts) == 2:
                                    lat = float(coords_parts[0].strip())
                                    lng = float(coords_parts[1].strip())
                                    existing_detected[customer_name] = (lat, lng)
                        except Exception as e:
                            logger.warning(f"Error parsing existing detected location row {idx}: {e}")
                            continue
            except Exception as e:
                logger.error(f"Error loading existing detected locations file: {e}")
                # Continue with empty dict if file is corrupted

        # Step 1: Clean up - remove customers that are now in master file
        customers_to_remove = []
        for customer_name in existing_detected:
            normalized_detected_name = self._normalize_name(customer_name)
            if normalized_detected_name in master_locations:
                customers_to_remove.append(customer_name)
                logger.info(f"Removing customer '{customer_name}' from detected file - normalized match found in master file")

        for customer_name in customers_to_remove:
            del existing_detected[customer_name]

        # Step 2: Add new API hits from current run
        for customer_name, coords in new_api_hits.items():
            existing_detected[customer_name] = coords
            logger.info(f"Added customer '{customer_name}' to detected file - found via API")

        # Step 3: Save updated detected locations back to Excel
        if existing_detected:
            # Convert to DataFrame
            rows = []
            for customer_name, (lat, lng) in existing_detected.items():
                coords_str = f"{lat:.6f},{lng:.6f}"  # Format with reasonable precision
                rows.append({
                    'שם לקוח \\ מחסן': customer_name,
                    'קואורדינטות': coords_str
                })

            detected_df = pd.DataFrame(rows)

            # Ensure assets directory exists
            os.makedirs("assets", exist_ok=True)

            # Save to Excel
            detected_df.to_excel(detected_file_path, index=False)
            logger.info(f"Saved {len(existing_detected)} detected locations to {detected_file_path}")
        else:
            # If no detected locations remain, remove the file
            if os.path.exists(detected_file_path):
                os.remove(detected_file_path)
                logger.info(f"Removed empty detected locations file: {detected_file_path}")
    
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
        Calculates Customer Force using: mean + (std * buffer) for route planning.
        """
        try:
            # Determine timeline (total days) from unique active dates in the file.
            total_days = 1
            if 'תאריך אספקה' in df.columns:
                date_series = pd.to_datetime(df['תאריך אספקה'], errors='coerce').dropna()
                if not date_series.empty:
                    total_days = date_series.dt.date.nunique()
                    if total_days < 1:
                        total_days = 1
                else:
                    total_days = 1

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
            
            # Step 2: Calculate Force (mean + std*buffer) including zero-days
            buffer = 1.0

            def calc_force_metrics(group):
                """Calculate force metrics for a customer group."""
                result = {
                    'force_quantity': self._calculate_client_force(group['כמות'].values, total_days, buffer)
                }
                
                if 'total_volume_m3' in group.columns:
                    result['force_volume'] = self._calculate_client_force(group['total_volume_m3'].values, total_days, buffer)
                else:
                    result['force_volume'] = 0.0
                
                return pd.Series(result)
            
            force_metrics = daily_orders.groupby('מס\' לקוח').apply(calc_force_metrics, include_groups=False).reset_index()
            
            logger.info(f"Calculated force metrics for {len(force_metrics)} customers using buffer={buffer} over total_days={total_days}")
            
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
            
            logger.info(f"Aggregated {len(df)} items to {len(aggregated)} customers using buffer={buffer} force over total_days={total_days}")
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            raise ValueError(f"שגיאה באגרגציה של הנתונים: {str(e)}")
    
    def recalculate_customer_force(self, aggregated_df: pd.DataFrame, raw_df: pd.DataFrame, buffer: float = None, buffer_multipliers: Optional[Dict[Any, float]] = None) -> pd.DataFrame:
        """
        Recalculate Customer Force metrics based on buffer multipliers.

        Args:
            aggregated_df: The aggregated customer dataframe (with existing force metrics)
            raw_df: The raw delivery data (before aggregation)
            buffer: Legacy single buffer value (deprecated, use buffer_multipliers)
            buffer_multipliers: Optional dict mapping customer IDs to specific buffer multipliers.
                               If not provided, uses Config.INITIAL_BUFFER_MULTIPLIER for all customers.

        Returns:
            Updated aggregated dataframe with new force metrics
        """
        try:
            # Determine timeline (total days) from unique active dates in the file.
            total_days = 1
            if 'תאריך אספקה' in raw_df.columns:
                date_series = pd.to_datetime(raw_df['תאריך אספקה'], errors='coerce').dropna()
                if not date_series.empty:
                    total_days = date_series.dt.date.nunique()
                    if total_days < 1:
                        total_days = 1
                else:
                    total_days = 1

            # Step 1: Daily Aggregation - Group by Customer ID and Date to get daily order sizes
            daily_group_cols = ['מס\' לקוח', 'תאריך אספקה']
            daily_agg_ops = {
                'כמות': 'sum'
            }
            if 'total_volume_m3' in raw_df.columns:
                daily_agg_ops['total_volume_m3'] = 'sum'
            
            daily_orders = raw_df.groupby(daily_group_cols, as_index=False).agg(daily_agg_ops)
            logger.info(f"Recalculating force: {len(daily_orders)} customer-date pairs")
            
            # Step 2: Calculate New Force (mean + std*buffer) including zero-days
            def calc_force_metrics(group):
                """Calculate force metrics for a customer group."""
                # Get customer ID from the group index (since we group by 'מס' לקוח')
                customer_id = group.name

                # Determine buffer multiplier for this customer
                if buffer_multipliers is not None and customer_id in buffer_multipliers:
                    customer_buffer = buffer_multipliers[customer_id]
                else:
                    customer_buffer = Config.INITIAL_BUFFER_MULTIPLIER

                # For backward compatibility, if buffer is provided, use it as fallback
                if buffer is not None and customer_buffer == Config.INITIAL_BUFFER_MULTIPLIER:
                    customer_buffer = buffer

                # Calculate force and standard deviation for quantity
                force_quantity, std_quantity = self._calculate_client_force(group['כמות'].values, total_days, customer_buffer)

                result = {
                    'force_quantity': force_quantity,
                    'std_quantity': std_quantity
                }

                if 'total_volume_m3' in group.columns:
                    force_volume, std_volume = self._calculate_client_force(group['total_volume_m3'].values, total_days, customer_buffer)
                    result['force_volume'] = force_volume
                    result['std_volume'] = std_volume
                else:
                    result['force_volume'] = 0.0
                    result['std_volume'] = 0.0

                return pd.Series(result)
            
            force_metrics = daily_orders.groupby('מס\' לקוח').apply(calc_force_metrics, include_groups=False).reset_index()
            
            if buffer_multipliers:
                unique_multipliers = set(buffer_multipliers.values())
                logger.info(f"Recalculated force metrics for {len(force_metrics)} customers using {len(unique_multipliers)} unique buffer multipliers over total_days={total_days}")
            else:
                default_buffer = Config.INITIAL_BUFFER_MULTIPLIER
                logger.info(f"Recalculated force metrics for {len(force_metrics)} customers using default buffer={default_buffer} over total_days={total_days}")
            
            # Step 3: Merge new force metrics into aggregated dataframe
            # Create a copy to avoid modifying the original
            updated_df = aggregated_df.copy()
            
            # Drop existing force columns if they exist (we'll replace them)
            if 'force_quantity' in updated_df.columns:
                updated_df = updated_df.drop(columns=['force_quantity'])
            if 'force_volume' in updated_df.columns:
                updated_df = updated_df.drop(columns=['force_volume'])
            if 'std_quantity' in updated_df.columns:
                updated_df = updated_df.drop(columns=['std_quantity'])
            if 'std_volume' in updated_df.columns:
                updated_df = updated_df.drop(columns=['std_volume'])

            # Merge on customer ID
            merge_cols = ['מס\' לקוח', 'force_quantity', 'force_volume', 'std_quantity', 'std_volume']
            updated_df = updated_df.merge(
                force_metrics[merge_cols],
                on='מס\' לקוח',
                how='left'
            )

            # Fill missing force values (shouldn't happen, but defensive)
            updated_df['force_quantity'] = updated_df['force_quantity'].fillna(0)
            updated_df['force_volume'] = updated_df['force_volume'].fillna(0)
            updated_df['std_quantity'] = updated_df['std_quantity'].fillna(0)
            updated_df['std_volume'] = updated_df['std_volume'].fillna(0)
            
            # Overwrite avg_quantity with the new force_quantity
            updated_df['avg_quantity'] = updated_df['force_quantity']
            
            if buffer_multipliers:
                logger.info(f"Updated customer force metrics with {len(buffer_multipliers)} custom buffer multipliers")
            else:
                default_buffer = Config.INITIAL_BUFFER_MULTIPLIER
                logger.info(f"Updated customer force metrics with default buffer={default_buffer}")
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
        """
        Add coordinates to customer data using hybrid geocoding approach.

        First checks master locations Excel file, then falls back to geocoding API.
        Tracks API-detected locations for building master file.
        """
        df_copy = df.copy()

        # Load master locations for hybrid geocoding
        master_locations = self._load_master_locations()
        api_calls_made = 0
        api_hits = {}  # Track customers found via API

        for idx, row in df_copy.iterrows():
            if pd.isna(row.get('lat')) or pd.isna(row.get('lng')):
                customer_name = str(row.get('שם לקוח', '')).strip()
                normalized_name = self._normalize_name(customer_name)

                # Step 1: Try master locations first (using normalized name)
                if normalized_name in master_locations:
                    lat, lng = master_locations[normalized_name]
                    df_copy.at[idx, 'lat'] = lat
                    df_copy.at[idx, 'lng'] = lng
                    logger.debug(f"Used master location for customer: {customer_name} (normalized: {normalized_name})")
                    continue

                # Step 2: Fall back to geocoding API
                address = row['כתובת']
                lat, lng = geo_service.get_coordinates(address)
                df_copy.at[idx, 'lat'] = lat
                df_copy.at[idx, 'lng'] = lng
                api_calls_made += 1

                # Track this customer as found via API
                if lat is not None and lng is not None:
                    api_hits[customer_name] = (lat, lng)

        logger.info(f"Geocoding completed: {len(master_locations)} master locations used, {api_calls_made} API calls made")

        # Update the API-detected locations file
        self._update_api_detected_file(api_hits, master_locations)

        return df_copy