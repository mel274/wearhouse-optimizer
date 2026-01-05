"""
Comparison service for performance analysis and historical backtesting.
Handles business logic for comparing optimized routes against actual performance.
"""
import logging
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import date

logger = logging.getLogger(__name__)


class ComparisonService:
    """Service for comparing optimized routes against historical actuals."""

    def parse_summary_file(self, file) -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[float], Optional[int]]:
        """
        Parse summary file and extract metrics.

        Args:
            file: Uploaded file object (CSV or Excel)

        Returns:
            Tuple of (df_summary, total_km, total_hours, total_days)
        """
        logger.info("Parsing summary file")

        df_summary = None
        summary_total_km = None
        summary_total_hours = None
        total_days_in_summary = None

        try:
            # Load file based on extension
            if file.name.endswith('.csv'):
                df_summary = pd.read_csv(file)
            else:
                df_summary = pd.read_excel(file)

            # Extract total kilometers
            if 'סהכ קמ' in df_summary.columns:
                summary_total_km = df_summary['סהכ קמ'].sum()
            elif 'Total KM' in df_summary.columns:
                summary_total_km = df_summary['Total KM'].sum()

            # Extract total hours
            if 'סהכ זמן נסיעה' in df_summary.columns:
                summary_total_hours = df_summary['סהכ זמן נסיעה'].sum()
            elif 'Total Driving Time' in df_summary.columns:
                summary_total_hours = df_summary['Total Driving Time'].sum()

            # Calculate total days from date range if available
            date_range_col = 'טווח תאריכים'
            if date_range_col in df_summary.columns:
                date_range_str = df_summary[date_range_col].iloc[0]
                if pd.notna(date_range_str):
                    date_range_str = str(date_range_str).strip()
                    if date_range_str.startswith('[') and date_range_str.endswith(']'):
                        date_range_str = date_range_str[1:-1].strip()
                    date_parts = date_range_str.split('-')
                    if len(date_parts) == 2:
                        start_str = date_parts[0].strip().replace('.', '/')
                        end_str = date_parts[1].strip().replace('.', '/')
                        summary_start = pd.to_datetime(start_str, dayfirst=True, errors='coerce')
                        summary_end = pd.to_datetime(end_str, dayfirst=True, errors='coerce')
                        if pd.notna(summary_start) and pd.notna(summary_end):
                            total_days_in_summary = (summary_end - summary_start).days + 1

            logger.info(f"Summary file parsed: km={summary_total_km}, hours={summary_total_hours}, days={total_days_in_summary}")

        except Exception as e:
            logger.error(f"Failed to parse summary file: {e}")
            raise

        return df_summary, summary_total_km, summary_total_hours, total_days_in_summary

    def prepare_simulation_data(self, raw_data: pd.DataFrame, aggregated_data: pd.DataFrame,
                              start_date: date, end_date: date,
                              order_date_col: str = 'תאריך אספקה',
                              customer_id_col: str = "מס' לקוח") -> pd.DataFrame:
        """
        Prepare historical data for simulation by filtering dates and merging coordinates.

        Args:
            raw_data: Raw historical order data
            aggregated_data: Aggregated data with geocoded coordinates
            start_date: Start date for filtering
            end_date: End date for filtering
            order_date_col: Column name for order dates
            customer_id_col: Column name for customer IDs

        Returns:
            Filtered and geocoded DataFrame ready for simulation
        """
        logger.info(f"Preparing simulation data for date range {start_date} to {end_date}")

        # Filter by date range
        filtered_data = raw_data.copy()
        filtered_data[order_date_col] = pd.to_datetime(filtered_data[order_date_col], dayfirst=True, errors='coerce')
        filtered_data = filtered_data[filtered_data[order_date_col].notna()]

        filtered_data = filtered_data[
            (filtered_data[order_date_col].dt.date >= start_date) &
            (filtered_data[order_date_col].dt.date <= end_date)
        ].copy()

        if len(filtered_data) == 0:
            raise ValueError("No orders found in the selected date range")

        # Merge coordinates from aggregated data
        if customer_id_col in aggregated_data.columns and customer_id_col in filtered_data.columns:
            # Create mapping of customer ID to coordinates
            coord_map = {}
            for idx, row in aggregated_data.iterrows():
                customer_id = row.get(customer_id_col)
                if pd.notna(customer_id) and pd.notna(row.get('lat')) and pd.notna(row.get('lng')):
                    coord_map[customer_id] = {
                        'lat': row['lat'],
                        'lng': row['lng']
                    }

            # Merge coordinates into filtered data
            filtered_data['lat'] = filtered_data[customer_id_col].map(
                lambda x: coord_map.get(x, {}).get('lat') if pd.notna(x) else None
            )
            filtered_data['lng'] = filtered_data[customer_id_col].map(
                lambda x: coord_map.get(x, {}).get('lng') if pd.notna(x) else None
            )
        else:
            raise ValueError("Could not merge coordinates. Customer ID column may be missing.")

        # Filter out rows without coordinates
        filtered_data_with_coords = filtered_data[
            filtered_data[['lat', 'lng']].notna().all(axis=1)
        ].copy()

        if len(filtered_data_with_coords) == 0:
            raise ValueError("No orders with valid coordinates found. Please ensure your data has been geocoded.")

        logger.info(f"Prepared {len(filtered_data_with_coords)} records for simulation")
        return filtered_data_with_coords

    def calculate_comparison_metrics(self, simulation_results: pd.DataFrame, filtered_data: pd.DataFrame,
                                   summary_total_km: Optional[float], summary_total_hours: Optional[float],
                                   total_days_in_summary: Optional[int], num_selected_days: int,
                                   start_date: date, end_date: date,
                                   order_date_col: str = 'תאריך אספקה') -> Dict[str, Any]:
        """
        Calculate comparison metrics between projected actuals and simulated optimized routes.

        Args:
            simulation_results: Results from historical simulation
            filtered_data: Filtered historical data
            summary_total_km: Total km from summary file
            summary_total_hours: Total hours from summary file
            total_days_in_summary: Number of days in summary period
            num_selected_days: Number of days in selected period
            start_date: Start date for analysis
            end_date: End date for analysis
            order_date_col: Column name for order dates

        Returns:
            Dictionary with comparison data and metrics
        """
        logger.info("Calculating comparison metrics")

        # Filter simulation results by date range
        simulation_results['Date'] = pd.to_datetime(simulation_results['Date'])
        filtered_simulation = simulation_results[
            (simulation_results['Date'].dt.date >= start_date) &
            (simulation_results['Date'].dt.date <= end_date)
        ].copy()

        # Calculate daily averages from summary file
        avg_actual_km_per_day = None
        avg_actual_time_per_day = None
        if summary_total_km is not None and total_days_in_summary is not None and total_days_in_summary > 0:
            avg_actual_km_per_day = summary_total_km / total_days_in_summary
        if summary_total_hours is not None and total_days_in_summary is not None and total_days_in_summary > 0:
            avg_actual_time_per_day = summary_total_hours / total_days_in_summary

        # Calculate projected and simulated totals
        projected_actual_km = avg_actual_km_per_day * num_selected_days if avg_actual_km_per_day else None
        projected_actual_time = avg_actual_time_per_day * num_selected_days if avg_actual_time_per_day else None

        simulated_km = filtered_simulation['Total_Distance_km'].sum()
        simulated_time = filtered_simulation['Max_Shift_Duration_hours'].sum()

        # Fleet utilization comparison
        system_avg_trucks = filtered_simulation['Active_Routes'].mean()

        # Historical metric: count unique route names per day
        historical_avg_trucks = None
        route_name_col = 'שם קו'

        if route_name_col in filtered_data.columns:
            filtered_data['Date'] = pd.to_datetime(filtered_data[order_date_col], dayfirst=True, errors='coerce')
            historical_for_selected = filtered_data[
                (filtered_data['Date'].dt.date >= start_date) &
                (filtered_data['Date'].dt.date <= end_date)
            ].copy()

            if len(historical_for_selected) > 0:
                daily_route_counts = historical_for_selected.groupby(
                    historical_for_selected['Date'].dt.date
                )[route_name_col].nunique()

                if len(daily_route_counts) > 0:
                    historical_avg_trucks = daily_route_counts.mean()

        # Build comparison table data
        comparison_data = []

        if projected_actual_km is not None:
            km_diff = simulated_km - projected_actual_km
            comparison_data.append({
                'Metric': 'Distance (km)',
                'Projected Actual': f"{projected_actual_km:,.1f}",
                'Simulated (Optimized)': f"{simulated_km:,.1f}",
                'Difference': f"{km_diff:+,.1f}"
            })

        if projected_actual_time is not None:
            time_diff = simulated_time - projected_actual_time
            comparison_data.append({
                'Metric': 'Time (hours)',
                'Projected Actual': f"{projected_actual_time:,.1f}",
                'Simulated (Optimized)': f"{simulated_time:,.1f}",
                'Difference': f"{time_diff:+,.1f}"
            })

        return {
            'system_avg_trucks': system_avg_trucks,
            'historical_avg_trucks': historical_avg_trucks,
            'comparison_data': comparison_data,
            'num_selected_days': num_selected_days
        }
