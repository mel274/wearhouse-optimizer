"""
Metrics and cost calculation logic.
"""
from typing import List, Any
from .types import MatrixData

def get_matrix_val(matrix: List[List[float]], i: int, j: int) -> float:
    """Safe retrieval from matrix."""
    if matrix is None or i >= len(matrix) or j >= len(matrix[i]):
        return float('inf')
    val = matrix[i][j]
    return val if val is not None else float('inf')

def solve_tsp_local(nodes: List[int], matrix: List[List[float]]) -> List[int]:
    """Nearest Neighbor heuristic for organizing a route sequence internally."""
    if not nodes: return []
    current = nodes[0]
    unvisited = set(nodes[1:])
    path = []
    while unvisited:
        # Find nearest neighbor
        next_node = min(unvisited, key=lambda x: get_matrix_val(matrix, current, x))
        unvisited.remove(next_node)
        path.append(next_node)
        current = next_node
    return path

def calc_approx_route_dist(route_indices: List[int], dist_matrix: List[List[float]]) -> float:
    """Calculates approximate distance of a route using NN TSP."""
    if not route_indices: return 0.0
    # Add warehouse (0) start and end
    ordered = [0] + solve_tsp_local([0] + route_indices, dist_matrix) + [0]
    total = 0.0
    for i in range(len(ordered) - 1):
        total += get_matrix_val(dist_matrix, ordered[i], ordered[i+1])
    return total

def calc_approx_route_duration(route_indices: List[int], dur_matrix: List[List[float]], 
                               service_time: int) -> float:
    """Calculates approximate duration of a route including service times."""
    if not route_indices: return 0.0
    ordered = [0] + solve_tsp_local([0] + route_indices, dur_matrix) + [0]
    total = 0.0
    for i in range(len(ordered) - 1):
        total += get_matrix_val(dur_matrix, ordered[i], ordered[i+1])
    return total + (len(route_indices) * service_time)

def calc_removal_savings(node: int, route: List[int], dist_matrix: List[List[float]]) -> float:
    """
    Approximates distance saved by removing a node from a route.
    Logic: (Prev->Node + Node->Next) - (Prev->Next)
    """
    if len(route) <= 1: 
        return get_matrix_val(dist_matrix, 0, node) * 2 # Out and back
    
    ordered = [0] + solve_tsp_local([0] + route, dist_matrix) + [0]
    try:
        idx = ordered.index(node)
        prev_node = ordered[idx-1]
        next_node = ordered[idx+1]
        
        old_segment = get_matrix_val(dist_matrix, prev_node, node) + get_matrix_val(dist_matrix, node, next_node)
        new_segment = get_matrix_val(dist_matrix, prev_node, next_node)
        
        return old_segment - new_segment
    except ValueError:
        return 0.0

def calc_insertion_cost(node: int, route: List[int], dist_matrix: List[List[float]]) -> float:
    """
    Calculates the cheapest cost to insert a node into a route.
    Logic: Min( (Prev->Node + Node->Next) - (Prev->Next) )
    """
    if not route: 
        return get_matrix_val(dist_matrix, 0, node) * 2
    
    ordered = [0] + solve_tsp_local([0] + route, dist_matrix) + [0]
    best_cost = float('inf')
    
    for i in range(len(ordered) - 1):
        u = ordered[i]
        v = ordered[i+1]
        
        d_u_node = get_matrix_val(dist_matrix, u, node)
        d_node_v = get_matrix_val(dist_matrix, node, v)
        d_u_v = get_matrix_val(dist_matrix, u, v)
        
        cost = (d_u_node + d_node_v) - d_u_v
        if cost < best_cost:
            best_cost = cost
    return best_cost