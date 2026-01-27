"""
Cluster service for route-name based initial solution seeding.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from config import Config
from shared.geo_utils import identify_small_truck_customers

logger = logging.getLogger(__name__)


class ClusterService:
    """
    Creates initial route assignments from existing route names (שם קו).
    Used for seeding OR-Tools solver with balanced starting routes.

    Note: This is a "best-effort" seed. Capacity/time constraints are NOT checked.
    If the seed is invalid, OR-Tools will reject it and fallback to default heuristic.
    """

    @staticmethod
    def get_small_truck_customers(coords: List[Tuple[float, float]]) -> Set[int]:
        """
        Get small truck zone customer indices using shared helper.
        Delegates to shared/geo_utils.py to avoid drift.
        """
        return identify_small_truck_customers(coords)

    def create_initial_routes_from_route_names(
        self,
        customer_data: pd.DataFrame,
        coords: List[Tuple[float, float]],
        route_name_col: str = 'current_route'
    ) -> Dict[str, List[int]]:
        """
        Group customers by their existing route names.

        Args:
            customer_data: DataFrame with customer data (must have route_name_col)
            coords: Full matrix_coords list where index 0 is depot, 1+ are customers
            route_name_col: Column containing route names

        Returns:
            Dictionary mapping route_name -> list of matrix indices (1-based, no depot)

        Note:
            - Index 0 in coords is the depot (skipped)
            - Customer at DataFrame position i maps to matrix index i+1
            - Uses enumerate() for safe positional mapping (handles non-standard indices)
        """
        clusters = {}

        # Use enumerate() for safe positional indexing (avoids get_loc issues with non-unique indices)
        for position, (_, row) in enumerate(customer_data.iterrows()):
            # Matrix index = position + 1 (depot is index 0)
            matrix_index = position + 1

            route_name = str(row.get(route_name_col, '')).strip()
            if not route_name or route_name == 'nan':
                route_name = '__unassigned__'

            if route_name not in clusters:
                clusters[route_name] = []
            clusters[route_name].append(matrix_index)

        logger.info(f"Created {len(clusters)} clusters from route names")
        return clusters

    def balance_clusters(
        self,
        clusters: Dict[str, List[int]],
        coords: List[Tuple[float, float]],
        min_cluster_size: int = 3,
        small_truck_indices: Set[int] = None
    ) -> List[List[int]]:
        """
        Balance clusters by merging small ones (< min_cluster_size)
        into nearest centroid cluster.

        Args:
            clusters: Dictionary mapping route_name -> list of matrix indices
            coords: Full matrix_coords (index 0 = depot)
            min_cluster_size: Minimum customers per cluster
            small_truck_indices: Set of node indices restricted to Small Trucks

        Returns:
            List of balanced routes (each route is list of customer matrix indices, no depot)
        """
        if small_truck_indices is None:
            small_truck_indices = set()

        # Convert to list of (route_name, indices)
        cluster_list = [(name, indices) for name, indices in clusters.items()]

        # Calculate centroids for each cluster
        def calc_centroid(indices: List[int]) -> Tuple[float, float]:
            if not indices:
                return (0.0, 0.0)
            lats = [coords[i][0] for i in indices]
            lngs = [coords[i][1] for i in indices]
            return (sum(lats) / len(lats), sum(lngs) / len(lngs))

        # Separate small and large clusters
        small_clusters = []
        large_clusters = []

        for name, indices in cluster_list:
            if len(indices) < min_cluster_size:
                small_clusters.append((name, indices))
            else:
                large_clusters.append((name, indices))

        # If no large clusters, just return all clusters as-is
        if not large_clusters:
            logger.warning("No large clusters found, returning all clusters as-is")
            return [indices for _, indices in cluster_list if indices]

        # Merge small clusters into nearest large cluster
        large_centroids = [[name, indices, calc_centroid(indices)] for name, indices in large_clusters]

        for small_name, small_indices in small_clusters:
            if not small_indices:
                continue

            small_centroid = calc_centroid(small_indices)

            # Find nearest large cluster
            min_dist = float('inf')
            nearest_idx = 0

            for i, (_, _, centroid) in enumerate(large_centroids):
                dist = ((small_centroid[0] - centroid[0])**2 +
                        (small_centroid[1] - centroid[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i

            # Merge into nearest
            large_centroids[nearest_idx][1].extend(small_indices)
            logger.debug(f"Merged small cluster '{small_name}' ({len(small_indices)} customers) into nearest cluster")

        # Return just the indices lists
        balanced = [indices for _, indices, _ in large_centroids]
        logger.info(f"Balanced clusters: {len(balanced)} clusters after merging small ones")
        return balanced

    def build_initial_routes_for_ortools(
        self,
        clusters: List[List[int]],
        num_vehicles: int,
        small_truck_vehicle_indices: List[int],
        small_truck_indices: Set[int]
    ) -> Optional[List[List[int]]]:
        """
        Convert clusters to OR-Tools route format, respecting vehicle constraints.

        CRITICAL: Routes must NOT include depot nodes. OR-Tools handles depot automatically.

        Assigns clusters with small truck zone customers to Small Truck vehicle indices.
        If more restricted clusters than Small Trucks, returns None to trigger fallback.

        Note: This is a "best-effort" seed. Capacity/time are NOT validated here.
        OR-Tools will reject invalid seeds and fallback to default heuristic.

        Args:
            clusters: List of routes (each is list of customer matrix indices, no depot)
            num_vehicles: Total number of vehicles
            small_truck_vehicle_indices: Vehicle indices that are Small Trucks
            small_truck_indices: Customer indices restricted to Small Trucks

        Returns:
            List of routes indexed by vehicle_id, or None if constraints cannot be satisfied
            Each route is a list of customer indices (NO DEPOT - OR-Tools adds depot automatically)
        """
        # Separate clusters by whether they contain small truck zone customers
        small_truck_clusters = []
        regular_clusters = []

        for cluster in clusters:
            has_restricted = any(idx in small_truck_indices for idx in cluster)
            if has_restricted:
                small_truck_clusters.append(cluster)
            else:
                regular_clusters.append(cluster)

        logger.info(f"Cluster split: {len(small_truck_clusters)} small truck zone clusters, {len(regular_clusters)} regular clusters")

        # Check if we have enough Small Trucks for restricted clusters
        if len(small_truck_clusters) > len(small_truck_vehicle_indices):
            logger.warning(f"Cannot satisfy small truck zone constraints: {len(small_truck_clusters)} clusters need Small Trucks, only {len(small_truck_vehicle_indices)} available")
            return None

        # Build routes array indexed by vehicle_id
        routes = [[] for _ in range(num_vehicles)]

        # Assign small truck zone clusters to Small Truck vehicles
        for i, cluster in enumerate(small_truck_clusters):
            vehicle_id = small_truck_vehicle_indices[i]
            routes[vehicle_id] = cluster  # NO DEPOT - just customer indices

        # Assign regular clusters to remaining vehicles
        big_truck_indices = [i for i in range(num_vehicles) if i not in small_truck_vehicle_indices]
        remaining_small = small_truck_vehicle_indices[len(small_truck_clusters):]
        available_vehicles = big_truck_indices + remaining_small

        # Log warning if more clusters than available vehicles
        total_clusters = len(small_truck_clusters) + len(regular_clusters)
        if len(regular_clusters) > len(available_vehicles):
            dropped_count = len(regular_clusters) - len(available_vehicles)
            logger.warning(f"More clusters ({total_clusters}) than vehicles ({num_vehicles}). Dropping {dropped_count} clusters from seed.")

        for i, cluster in enumerate(regular_clusters):
            if i < len(available_vehicles):
                vehicle_id = available_vehicles[i]
                routes[vehicle_id] = cluster  # NO DEPOT - just customer indices

        non_empty_routes = len([r for r in routes if r])
        logger.info(f"Built initial routes: {non_empty_routes} non-empty routes for {num_vehicles} vehicles")
        return routes
