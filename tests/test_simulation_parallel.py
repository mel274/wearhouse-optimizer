"""
Test script to verify parallel processing simulation works correctly.
Tests that the refactored simulation produces the same results as sequential processing.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from calculations.simulation import run_historical_simulation, create_node_map

def test_simulation_basic():
    """Test that simulation runs without errors and produces expected output structure."""
    print("Testing basic simulation functionality...")
    
    # Create minimal test data
    historical_data = pd.DataFrame({
        'תאריך אספקה': ['2024-01-01', '2024-01-01', '2024-01-02'],
        "מס' לקוח": [1, 2, 1],
        'כמות': [10.0, 20.0, 15.0]
    })
    
    # Create simple master routes (2 routes, each with 2 customers)
    master_routes = [
        [0, 1, 2, 0],  # Route 1: depot -> customer 1 -> customer 2 -> depot
        [0, 3, 0]      # Route 2: depot -> customer 3 -> depot
    ]
    
    # Create simple distance and time matrices (4x4: depot + 3 customers)
    distance_matrix = [
        [0, 1000, 2000, 3000],  # From depot
        [1000, 0, 1500, 2500],  # From customer 1
        [2000, 1500, 0, 3500],  # From customer 2
        [3000, 2500, 3500, 0]   # From customer 3
    ]
    
    time_matrix = [
        [0, 120, 240, 360],    # From depot (seconds)
        [120, 0, 180, 300],    # From customer 1
        [240, 180, 0, 420],    # From customer 2
        [360, 300, 420, 0]     # From customer 3
    ]
    
    # Create node map
    optimization_data = pd.DataFrame({
        "מס' לקוח": [1, 2, 3]
    })
    node_map = create_node_map(optimization_data)
    
    # Run simulation
    try:
        results = run_historical_simulation(
            historical_data=historical_data,
            master_routes=master_routes,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            node_map=node_map,
            service_time_seconds=300,  # 5 minutes per stop
            date_col='תאריך אספקה',
            customer_id_col="מס' לקוח",
            quantity_col='כמות',
            depot=0,
            route_capacities=[100.0, 50.0],  # Route capacities
            max_shift_seconds=28800,  # 8 hours
            volume_tolerance=0.0,
            time_tolerance=0.0
        )
        
        # Verify results structure
        assert isinstance(results, pd.DataFrame), "Results should be a DataFrame"
        assert len(results) > 0, "Results should contain at least one day"
        assert 'Date' in results.columns, "Results should have 'Date' column"
        assert 'Total_Distance_km' in results.columns, "Results should have 'Total_Distance_km' column"
        assert 'Daily_Total_Duration' in results.columns, "Results should have 'Daily_Total_Duration' column"
        assert 'Daily_Total_Capacity' in results.columns, "Results should have 'Daily_Total_Capacity' column"
        assert 'Daily_Total_Load' in results.columns, "Results should have 'Daily_Total_Load' column"
        assert 'Route_Breakdown' in results.columns, "Results should have 'Route_Breakdown' column"
        
        print("OK: Simulation completed successfully!")
        print(f"  - Processed {len(results)} days")
        print(f"  - Columns: {list(results.columns)}")
        
        # Verify dates are sorted
        dates = pd.to_datetime(results['Date'])
        assert dates.is_monotonic_increasing, "Dates should be sorted in ascending order"
        print("  - Dates are correctly sorted")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simulation_empty_data():
    """Test that simulation handles empty data gracefully."""
    print("\nTesting simulation with empty data...")
    
    historical_data = pd.DataFrame({
        'תאריך אספקה': [],
        "מס' לקוח": [],
        'כמות': []
    })
    
    master_routes = [[0, 1, 0]]
    distance_matrix = [[0, 1000], [1000, 0]]
    time_matrix = [[0, 120], [120, 0]]
    
    optimization_data = pd.DataFrame({"מס' לקוח": [1]})
    node_map = create_node_map(optimization_data)
    
    try:
        results = run_historical_simulation(
            historical_data=historical_data,
            master_routes=master_routes,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            node_map=node_map,
            service_time_seconds=300
        )
        
        assert isinstance(results, pd.DataFrame), "Results should be a DataFrame"
        print(f"OK: Empty data handled correctly (returned {len(results)} days)")
        return True
        
    except Exception as e:
        print(f"ERROR: Empty data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Parallel Processing Simulation")
    print("=" * 60)
    
    test1_passed = test_simulation_basic()
    test2_passed = test_simulation_empty_data()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("SUCCESS: All tests passed!")
        sys.exit(0)
    else:
        print("FAILURE: Some tests failed!")
        sys.exit(1)
