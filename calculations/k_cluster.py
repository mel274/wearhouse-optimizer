"""
Initial customer clustering using K-Means.
"""
import math
import logging
from typing import List
from .types import Cluster, Coords

logger = logging.getLogger(__name__)

def perform_clustering(customer_coords: List[Coords], 
                       customer_demands: List[int],
                       fleet_size: int, 
                       capacity: int) -> List[Cluster]:
    """
    Groups customers into initial clusters based on location and demand.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        logger.error("scikit-learn not found. Clustering will fail.")
        return []

    # 1. Determine K (Number of Trucks)
    total_demand = sum(customer_demands)
    min_k_by_volume = int(math.ceil(total_demand / capacity))
    min_k = max(min_k_by_volume, 1)
    
    # We allow some slack for the optimizer to work with
    # Start with a count between Min required and Max Fleet
    search_k = min(fleet_size, max(min_k + 1, int(len(customer_coords) / 4)))
    search_k = max(search_k, min_k)
    
    logger.info(f"Clustering: Demand requires {min_k_by_volume} trucks. Starting with K={search_k}.")

    if not customer_coords:
        return []

    # 2. Run K-Means
    kmeans = KMeans(n_clusters=search_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(customer_coords)
    
    # 3. Convert to Cluster objects
    # Note: indices must match the global coordinate list (0 is warehouse)
    # customer_coords[i] corresponds to global index i + 1
    clusters = [[] for _ in range(search_k)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx + 1)
        
    return [{'indices': c} for c in clusters if c]