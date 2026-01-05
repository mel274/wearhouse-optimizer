"""
Core business logic tests for the Warehouse Optimizer.
Tests critical functions to ensure mathematical accuracy and data validation.
"""
import sys
from unittest.mock import MagicMock

# MOCK DEPENDENCIES GLOBALLY BEFORE IMPORTING APP CODE
sys.modules["pandas"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["openrouteservice"] = MagicMock()

# Now we can safely import the app code
from calculations.logistics import calculate_fleet_metrics
from utils import validate_address, retry_with_backoff
from exceptions import DataValidationError

import unittest
from unittest.mock import Mock


class TestCalculateFleetMetrics(unittest.TestCase):
    """Test the calculate_fleet_metrics function for accurate truck calculations."""

    def test_big_truck_calculation(self):
        """Test that a known volume correctly calculates truck requirements using the real function."""
        # Create a mock DataFrame that behaves like the real pandas DataFrame
        mock_df = Mock()

        # Configure copy() to return a mock that supports iterrows and tracks column assignments
        mock_copied_df = Mock()
        assigned_columns = {}
        def track_assignment(self, key, value):
            assigned_columns[key] = value

        # Configure iterrows to return test data (outside Gush Dan bounds)
        mock_copied_df.iterrows.return_value = [
            (0, Mock(
                get=lambda key, default=None: {
                    'lat': 32.0, 'lng': 34.8, 'total_volume_m3': 100.0
                }.get(key, default)
            ))
        ]

        mock_copied_df.__setitem__ = track_assignment
        mock_df.copy.return_value = mock_copied_df

        fleet_settings = {
            'big_truck_vol': 32.0,
            'small_truck_vol': 27.0,
            'safety_factor': 0.7
        }

        # Call the real function
        result = calculate_fleet_metrics(mock_df, fleet_settings)

        # Verify the real function assigned the correct values
        # Expected: 100.0 / (32.0 * 0.7) = 100.0 / 22.4 = 4.46 -> ceil(4.46) = 5 trucks
        self.assertIn('truck_type', assigned_columns)
        self.assertIn('trucks_needed', assigned_columns)
        self.assertIn('color', assigned_columns)

        self.assertEqual(assigned_columns['truck_type'], ["Big Truck"])
        self.assertEqual(assigned_columns['trucks_needed'], [5])
        self.assertEqual(assigned_columns['color'], ["#0000FF"])

    def test_small_truck_gush_dan_calculation(self):
        """Test that locations in Gush Dan use small trucks using the real function."""
        mock_df = Mock()

        # Configure copy() to return a mock that supports iterrows and tracks column assignments
        mock_copied_df = Mock()
        assigned_columns = {}
        def track_assignment(self, key, value):
            assigned_columns[key] = value

        # Configure iterrows to return Gush Dan coordinates
        mock_copied_df.iterrows.return_value = [
            (0, Mock(
                get=lambda key, default=None: {
                    'lat': 32.08, 'lng': 34.82, 'total_volume_m3': 50.0  # Within Gush Dan bounds
                }.get(key, default)
            ))
        ]

        mock_copied_df.__setitem__ = track_assignment
        mock_df.copy.return_value = mock_copied_df

        fleet_settings = {
            'big_truck_vol': 32.0,
            'small_truck_vol': 27.0,
            'safety_factor': 0.7
        }

        # Call the real function
        result = calculate_fleet_metrics(mock_df, fleet_settings)

        # Expected: 50.0 / (27.0 * 0.7) = 50.0 / 18.9 = 2.65 -> ceil(2.65) = 3 trucks
        self.assertEqual(assigned_columns['truck_type'], ["Small Truck"])
        self.assertEqual(assigned_columns['trucks_needed'], [3])
        self.assertEqual(assigned_columns['color'], ["#FF0000"])

    def test_zero_volume_handling(self):
        """Test that zero volume doesn't cause division by zero using the real function."""
        mock_df = Mock()

        # Configure copy() to return a mock that supports iterrows and tracks column assignments
        mock_copied_df = Mock()
        assigned_columns = {}
        def track_assignment(self, key, value):
            assigned_columns[key] = value

        # Configure iterrows to return zero volume data
        mock_copied_df.iterrows.return_value = [
            (0, Mock(
                get=lambda key, default=None: {
                    'lat': 32.0, 'lng': 34.8, 'total_volume_m3': 0.0
                }.get(key, default)
            ))
        ]

        mock_copied_df.__setitem__ = track_assignment
        mock_df.copy.return_value = mock_copied_df

        fleet_settings = {
            'big_truck_vol': 32.0,
            'small_truck_vol': 27.0,
            'safety_factor': 0.7
        }

        result = calculate_fleet_metrics(mock_df, fleet_settings)

        self.assertEqual(assigned_columns['trucks_needed'], [0])

    def test_exact_capacity_match(self):
        """Test edge case where volume exactly matches effective capacity using the real function."""
        mock_df = Mock()

        # Configure copy() to return a mock that supports iterrows and tracks column assignments
        mock_copied_df = Mock()
        assigned_columns = {}
        def track_assignment(self, key, value):
            assigned_columns[key] = value

        # Configure iterrows to return exact capacity match data
        mock_copied_df.iterrows.return_value = [
            (0, Mock(
                get=lambda key, default=None: {
                    'lat': 32.0, 'lng': 34.8, 'total_volume_m3': 22.4  # Exactly matches effective capacity
                }.get(key, default)
            ))
        ]

        mock_copied_df.__setitem__ = track_assignment
        mock_df.copy.return_value = mock_copied_df

        fleet_settings = {
            'big_truck_vol': 32.0,
            'small_truck_vol': 27.0,
            'safety_factor': 0.7
        }

        result = calculate_fleet_metrics(mock_df, fleet_settings)

        # Should require exactly 1 truck
        self.assertEqual(assigned_columns['trucks_needed'], [1])


class TestValidateAddress(unittest.TestCase):
    """Test the validate_address function for proper sanitization."""

    def test_sanitizes_malicious_input(self):
        """Test that dangerous characters are properly removed using the real function."""
        dirty_address = '<script>alert("xss")</script>Bad Address"; DROP TABLE users;'

        result = validate_address(dirty_address)

        # Should remove all dangerous characters
        self.assertNotIn('<', result)
        self.assertNotIn('>', result)
        self.assertNotIn('"', result)
        self.assertNotIn(';', result)
        self.assertIn('Bad Address', result)  # But keep the legitimate part

    def test_preserves_legitimate_characters(self):
        """Test that normal address characters are preserved using the real function."""
        normal_address = "123 Main St, Tel Aviv, Israel"

        result = validate_address(normal_address)

        self.assertEqual(result, normal_address)

    def test_handles_whitespace(self):
        """Test that whitespace is properly handled using the real function."""
        address_with_spaces = "  123 Main St  "

        result = validate_address(address_with_spaces)

        self.assertEqual(result, "123 Main St")  # Should strip whitespace

    def test_rejects_empty_string(self):
        """Test that empty strings raise validation errors using the real function."""
        with self.assertRaises(DataValidationError) as cm:
            validate_address("")
        self.assertIn("Address must be a non-empty string", str(cm.exception))

    def test_rejects_non_string_input(self):
        """Test that non-string inputs raise validation errors using the real function."""
        with self.assertRaises(DataValidationError) as cm:
            validate_address(123)
        self.assertIn("Address must be a non-empty string", str(cm.exception))

        with self.assertRaises(DataValidationError) as cm:
            validate_address(None)
        self.assertIn("Address must be a non-empty string", str(cm.exception))

    def test_rejects_too_short_after_sanitization(self):
        """Test that addresses become too short after sanitization using the real function."""
        with self.assertRaises(DataValidationError) as cm:
            validate_address("<>")  # This becomes empty string after sanitization
        self.assertIn("Address too short after sanitization", str(cm.exception))

    def test_rejects_too_long_address(self):
        """Test that extremely long addresses are rejected using the real function."""
        long_address = "A" * 201  # Longer than 200 character limit

        with self.assertRaises(DataValidationError) as cm:
            validate_address(long_address)
        self.assertIn("Address too long", str(cm.exception))


class TestRetryWithBackoff(unittest.TestCase):
    """Test the retry_with_backoff decorator for proper retry behavior."""

    def test_successful_first_attempt(self):
        """Test that function succeeds on first attempt without retries using the real decorator."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, delay=0.01)  # Very short delay for testing
        def dummy_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = dummy_function()

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 1)

    def test_retry_on_failure_then_success(self):
        """Test that function retries on failure and eventually succeeds using the real decorator."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, delay=0.01)  # Very short delay for testing
        def dummy_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first two attempts
                raise ValueError(f"Attempt {call_count} failed")
            return "success"

        result = dummy_function()

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)  # Should have been called 3 times

    def test_exhausts_all_attempts_and_fails(self):
        """Test that function fails after exhausting all retry attempts using the real decorator."""
        call_count = 0

        @retry_with_backoff(max_attempts=2, delay=0.01)  # Very short delay for testing
        def dummy_function():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Persistent failure on attempt {call_count}")

        with self.assertRaises(ValueError) as cm:
            dummy_function()

        self.assertIn("Persistent failure on attempt 2", str(cm.exception))
        self.assertEqual(call_count, 2)  # Should have been called exactly 2 times


if __name__ == '__main__':
    unittest.main()