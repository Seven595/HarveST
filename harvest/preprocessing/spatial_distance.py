import numpy as np
import pandas as pd
import numba
from typing import Optional, Tuple
import logging
import ot


@numba.njit
def euclid_dist(t1: np.ndarray, t2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    sum_sq = 0
    for i in range(t1.shape[0]):
        sum_sq += (t1[i] - t2[i])**2
    return np.sqrt(sum_sq)


@numba.njit(parallel=True)
def pairwise_distance(X: np.ndarray) -> np.ndarray:
    """Calculate pairwise distances between all points in X."""
    n = X.shape[0]
    dist_matrix = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            dist_matrix[i][j] = euclid_dist(X[i], X[j])
    return dist_matrix


class SpatialDistanceCalculator:
    """Calculate spatial distances between cells."""
    
    @staticmethod
    def calculate_p(adj: np.ndarray, l: float) -> Tuple[float, np.ndarray]:
        """Calculate percentage contribution of neighbor nodes."""
        adj = adj.astype(float)
        adj_exp = 1 - np.tanh((adj**2)/(2*(l**2)))
        adj_exp[adj_exp == 1] = 0
        
        p = np.mean(np.sum(adj_exp, 1))
        return p, adj_exp
    
    @staticmethod
    def search_l_cc(p: float, adj: np.ndarray, start: float = 0.01, end: float = 1000,
                   tol: float = 0.01, max_run: int = 100, 
                   logger: Optional[logging.Logger] = None) -> Optional[float]:
        """Search for optimal l value to achieve target p."""
        run = 0
        p_low, _ = SpatialDistanceCalculator.calculate_p(adj, start)
        p_high, _ = SpatialDistanceCalculator.calculate_p(adj, end)
        
        log_msg = lambda msg: logger.info(msg) if logger else print(msg)
        
        if p_low > p + tol:
            log_msg("l not found, try smaller start point.")
            return None
        elif p_high < p - tol:
            log_msg("l not found, try bigger end point.")
            return None
        elif np.abs(p_low - p) <= tol:
            log_msg(f"recommended l = {start}")
            return start
        elif np.abs(p_high - p) <= tol:
            log_msg(f"recommended l = {end}")
            return end
        
        while (p_low + tol) < p < (p_high - tol):
            run += 1
            log_msg(f"Run {run}: l [{start}, {end}], p [{p_low}, {p_high}]")
            
            if run > max_run:
                log_msg(f"Exact l not found, closest values are:\nl={start}: p={p_low}\nl={end}: p={p_high}")
                return None
            
            mid = (start + end) / 2
            p_mid, _ = SpatialDistanceCalculator.calculate_p(adj, mid)
            
            if np.abs(p_mid - p) <= tol:
                log_msg(f"recommended l = {mid}")
                return mid
            
            if p_mid <= p:
                start = mid
                p_low = p_mid
            else:
                end = mid
                p_high = p_mid
    
    @staticmethod
    def cal_cc_weight(adata, p: float = 1.0, start: float = 1, end: float = 100,
                     radius: float = 480, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
        """Calculate cell-cell weights based on spatial coordinates."""
        log_msg = lambda msg: logger.info(msg) if logger else print(msg)
        
        X = adata.obsm["spatial"]
        log_msg(f"Computing spatial distances for {X.shape[0]} cells")
        
        # Calculate distance matrix using numba acceleration
        dist_matrix = pairwise_distance(X)
        df_adj = pd.DataFrame(dist_matrix)
        
        # Threshold connections by radius
        df_adj[df_adj > radius] = 0
        
        # Calculate average number of neighboring cells
        non_zero_count = df_adj.apply(lambda row: row.astype(bool).sum(), axis=1)
        log_msg(f"Average number of neighboring cells per cell: {np.mean(non_zero_count)}")
        
        # Find optimal l value
        l = SpatialDistanceCalculator.search_l_cc(
            p, df_adj, start=start, end=end, tol=0.001, max_run=100, logger=logger)
        _, adj_exp = SpatialDistanceCalculator.calculate_p(df_adj, l)
        
        # Convert to DataFrame
        df_adj = pd.DataFrame(adj_exp)
        return df_adj 