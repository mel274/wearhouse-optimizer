"""
Utility functions for Warehouse Location & Route Optimizer.
Unified utilities from root utils.py and calculations/utils.py.
"""

import pandas as pd
from typing import Union, Optional, Callable, Any, List, Tuple
import io
import time
import logging
import re
import math
from functools import wraps
from config import Config
from exceptions import DataValidationError, APIRateLimitError

logger = logging.getLogger(__name__)

def retry_with_backoff(max_attempts: int = Config.API_RETRY_ATTEMPTS,
                      delay: float = Config.RATE_LIMIT_DELAY):
    """Decorator for retrying API calls with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logger.error(f"API call failed after {max_attempts} attempts: {e}")
                        raise e

                    wait_time = delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)

            raise last_exception
        return wrapper
    return decorator

def validate_address(address: str) -> str:
    """Validate and sanitize address input."""
    if not address or not isinstance(address, str):
        raise DataValidationError("Address must be a non-empty string")

    # Remove potentially harmful characters
    sanitized = re.sub(r'[<>"\';]', '', address.strip())

    if len(sanitized) < 2:
        raise DataValidationError("Address too short after sanitization")

    if len(sanitized) > 200:
        raise DataValidationError("Address too long")

    logger.debug(f"Address validated: '{sanitized[:50]}...'")
    return sanitized

def validate_coordinates(lat: float, lng: float) -> bool:
    """Validate coordinates are within configurable bounds."""
    try:
        lat_float, lng_float = float(lat), float(lng)
    except (ValueError, TypeError):
        return False

    return (Config.ISRAEL_LAT_MIN <= lat_float <= Config.ISRAEL_LAT_MAX and
            Config.ISRAEL_LNG_MIN <= lng_float <= Config.ISRAEL_LNG_MAX)


def validate_hebrew_encoding(file_buffer: Union[io.BytesIO, io.StringIO]) -> pd.DataFrame:
    """
    Validate and read Hebrew-encoded Excel/CSV files with proper UTF-8 handling.
    """
    logger.info("Starting Hebrew file encoding validation")

    try:
        # Try reading as Excel first
        if hasattr(file_buffer, 'name') and file_buffer.name.endswith(('.xlsx', '.xls')):
            logger.debug("Attempting to read as Excel file")
            df = pd.read_excel(file_buffer, engine='openpyxl')
            logger.info(f"Successfully read Excel file with {len(df)} rows")
        else:
            # Try different encodings for CSV files
            encodings = ['utf-8', 'utf-8-sig', 'cp1255', 'iso-8859-8']

            for encoding in encodings:
                try:
                    file_buffer.seek(0)  # Reset buffer position
                    logger.debug(f"Trying CSV encoding: {encoding}")
                    df = pd.read_csv(file_buffer, encoding=encoding)
                    logger.info(f"Successfully read CSV file with {encoding} encoding, {len(df)} rows")
                    break
                except UnicodeDecodeError:
                    logger.debug(f"Failed to read with {encoding} encoding")
                    continue
            else:
                error_msg = "Could not read file with any supported Hebrew encoding"
                logger.error(error_msg)
                raise DataValidationError(error_msg)

        # Validate Hebrew text encoding
        if df.empty:
            error_msg = "File is empty or could not be parsed"
            logger.error(error_msg)
            raise DataValidationError(error_msg)

        logger.info(f"File validation completed successfully. Found columns: {list(df.columns)}")
        return df

    except DataValidationError:
        raise
    except Exception as e:
        error_msg = f"Error reading Hebrew-encoded file: {str(e)}"
        logger.error(error_msg)
        raise DataValidationError(error_msg)


def format_time(seconds: Union[int, float]) -> str:
    """Convert seconds to HH:MM:SS format."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    """
    Decodes a Google-encoded polyline string into a list of (lat, lng) tuples.
    This fixes the issue where routes appear as straight lines.
    """
    index, lat, lng = 0, 0, 0
    coordinates = []
    length = len(polyline_str)

    try:
        while index < length:
            # Decode Latitude
            shift, result = 0, 0
            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if byte < 0x20: break
            dlat = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += dlat

            # Decode Longitude
            shift, result = 0, 0
            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if byte < 0x20: break
            dlng = ~(result >> 1) if (result & 1) else (result >> 1)
            lng += dlng

            coordinates.append((lat / 1e5, lng / 1e5))
        return coordinates
    except Exception as e:
        logger.error(f"Error decoding polyline: {e}")
        return []
