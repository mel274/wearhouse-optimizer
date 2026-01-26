# Fix: Revert Time Constraint to Soft for Master Route Planning

## Problem
Optimization phase fails because the hard time constraint (`max_shift_seconds`) makes the problem infeasible when planning master routes.

**Root Cause**: Master routes calculate time as if ALL customers on the route are visited every day. With 708 customers and 15-min service time each, the solver cannot fit all customers into routes that respect a hard 12-hour limit.

**Reality**: Master routes are territory assignments. On any given day, only a subset of customers on each route have orders. The simulation already handles this by creating sub-routes with only active customers.

## Solution
Use **SOFT** time constraints for master route planning, **HARD** enforcement in simulation.

### File to Modify
`calculations/route_optimizer.py`

### Current Code (lines 341-355)
```python
time_callback_index = routing.RegisterTransitCallback(time_callback)

# Hard time limit: routes cannot exceed max_shift_seconds
# The solver must always stay within the shift time limit set by the user
routing.AddDimension(
    time_callback_index,
    0,  # slack
    data['max_shift_seconds'],  # hard cap - uses max_shift_seconds from UI settings
    True,  # fix_start_cumul_to_zero
    "Time"
)

# Get the Time dimension and set global span cost to balance workloads across drivers
time_dim = routing.GetDimensionOrDie("Time")
time_dim.SetGlobalSpanCostCoefficient(100)
```

### New Code
```python
time_callback_index = routing.RegisterTransitCallback(time_callback)

# Master Route Planning: Use SOFT time constraints
# Master routes are territory assignments - they group customers into logical routes.
# The actual time constraint enforcement happens in simulation when testing daily sub-routes.
# Using soft constraints here allows the solver to find valid territory assignments
# even when the theoretical "all customers visited" time exceeds the shift limit.
routing.AddDimension(
    time_callback_index,
    0,  # slack
    172800,  # 48-hour horizon (soft constraint ceiling)
    True,  # fix_start_cumul_to_zero
    "Time"
)

# Apply soft upper bound for time - penalize exceeding max_shift_seconds
time_dimension = routing.GetDimensionOrDie("Time")
for vehicle_id in range(vehicle_limit):
    index = routing.End(vehicle_id)
    time_dimension.SetCumulVarSoftUpperBound(index, data['max_shift_seconds'], 1000)

# Set global span cost to balance workloads across drivers
time_dimension.SetGlobalSpanCostCoefficient(100)
```

## Key Changes
1. Change `AddDimension` max from `data['max_shift_seconds']` to `172800` (48 hours) - allows solver flexibility
2. Add `SetCumulVarSoftUpperBound` loop - penalizes (not blocks) routes exceeding `max_shift_seconds`
3. Rename `time_dim` to `time_dimension` for consistency

## Why This Works
| Phase | Constraint Type | Purpose |
|-------|----------------|---------|
| Master Route Planning | SOFT | Territory assignment - group customers logically |
| Simulation (daily) | HARD | Validates actual sub-routes against `max_shift_seconds` |

The simulation at `calculations/simulation.py:271` already enforces hard limits:
```python
time_ok = True if limit_time is None else (metrics['duration'] <= limit_time)
```

## Testing
1. Run optimization with 12-hour shift limit
2. Verify master routes are created successfully
3. Check simulation results for any "Overtime" violations on actual daily routes
