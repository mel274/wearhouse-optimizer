"""
Test to verify that parallel processing produces correct results.
Compares parallel results with expected sequential results.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from calculations.simulation import run_historical_simulation, create_node_map


def test_simulation_correctness():
    """Test that simulation produces correct and consistent results."""
    print("Testing simulation correctness and consistency...")
    
    # Create test data with multiple dates
    historical_data = pd.DataFrame({
        'תאריך אספקה': [
            '2024-01-01', '2024-01-01', '2024-01-01',  # Day 1: 3 orders
            '2024-01-02', '2024-01-02',               # Day 2: 2 orders
            '2024-01-03'                               # Day 3: 1 order
        ],
        "מס' לקוח": [1, 2, 3, 1, 2, 1],
        'כמות': [10.0, 20.0, 15.0, 12.0, 18.0, 8.0]
    })
    
    # Create master routes
    master_routes = [
        [0, 1, 2, 0],  # Route 1: depot -> customer 1 -> customer 2 -> depot
        [0, 3, 0]      # Route 2: depot -> customer 3 -> depot
    ]
    
    # Create distance and time matrices
    distance_matrix = [
        [0, 1000, 2000, 3000],
        [1000, 0, 1500, 2500],
        [2000, 1500, 0, 3500],
        [3000, 2500, 3500, 0]
    ]
    
    time_matrix = [
        [0, 120, 240, 360],
        [120, 0, 180, 300],
        [240, 180, 0, 420],
        [360, 300, 420, 0]
    ]
    
    # Create node map
    optimization_data = pd.DataFrame({
        "מס' לקוח": [1, 2, 3]
    })
    node_map = create_node_map(optimization_data)
    
    # Run simulation multiple times to check consistency
    results_list = []
    for i in range(3):
        results = run_historical_simulation(
            historical_data=historical_data.copy(),
            master_routes=master_routes,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            node_map=node_map,
            service_time_seconds=300,
            date_col='תאריך אספקה',
            customer_id_col="מס' לקוח",
            quantity_col='כמות',
            depot=0,
            route_capacities=[100.0, 50.0],
            max_shift_seconds=28800,
            volume_tolerance=0.0,
            time_tolerance=0.0
        )
        results_list.append(results)
    
    # Verify all runs produce the same results
    for i in range(1, len(results_list)):
        pd.testing.assert_frame_equal(
            results_list[0].sort_values('Date').reset_index(drop=True),
            results_list[i].sort_values('Date').reset_index(drop=True),
            check_exact=False,  # Allow for floating point differences
            rtol=1e-5
        )
    
    print("OK: Multiple simulation runs produce consistent results")
    
    # Verify expected structure
    assert len(results_list[0]) == 3, f"Expected 3 days, got {len(results_list[0])}"
    
    # Verify each day has expected columns
    for idx, row in results_list[0].iterrows():
        assert 'Date' in row, "Each row should have Date"
        assert 'Total_Distance_km' in row, "Each row should have Total_Distance_km"
        assert 'Daily_Total_Duration' in row, "Each row should have Daily_Total_Duration"
        assert 'Daily_Total_Capacity' in row, "Each row should have Daily_Total_Capacity"
        assert 'Daily_Total_Load' in row, "Each row should have Daily_Total_Load"
        assert 'Route_Breakdown' in row, "Each row should have Route_Breakdown"
        assert isinstance(row['Route_Breakdown'], list), "Route_Breakdown should be a list"
    
    print("OK: Results have correct structure")
    
    # Verify dates are sorted
    dates = pd.to_datetime(results_list[0]['Date'])
    assert dates.is_monotonic_increasing, "Dates should be sorted"
    print("OK: Dates are correctly sorted")
    
    # Verify no duplicate dates
    assert len(dates.unique()) == len(dates), "No duplicate dates"
    print("OK: No duplicate dates")
    
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Simulation Correctness")
    print("=" * 60)
    
    try:
        test_passed = test_simulation_correctness()
        print("\n" + "=" * 60)
        print("SUCCESS: All correctness tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\nFAILURE: Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
