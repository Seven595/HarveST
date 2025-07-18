import numpy as np
import numba
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed
import time
from typing import Optional, Tuple
import logging


class MutualInformationCalculator:
    """Calculate mutual information between genes."""
    
    @staticmethod
    def generate_mi(gene_data_array: np.ndarray, logger: Optional[logging.Logger] = None) -> np.ndarray:
        """Calculate mutual information matrix using original V1 logic."""
        N = gene_data_array.shape[0]
        mi_matrix = np.zeros((N, N))
        
        if logger:
            logger.info(f"Computing mutual information matrix for {N} genes (using V1 logic)")
        
        for i in tqdm(range(N)):
            for j in range(i+1, N):
                if i != j:
                    X = gene_data_array[i].reshape(-1, 1)
                    y = gene_data_array[j]
                    mi = mutual_info_regression(X, y, random_state=2023)
                    mi_matrix[i, j] = mi[0]
                    mi_matrix[j, i] = mi[0]
        
        return mi_matrix
    
    @staticmethod
    def _compute_mi_for_pair(args: Tuple) -> Tuple[int, int, float]:
        """Compute mutual information for a single gene pair."""
        i, j, X, y = args
        mi = mutual_info_regression(X.reshape(-1, 1), y, random_state=2023)
        return (i, j, mi[0])
    
    @staticmethod
    def generate_mi_parallel(gene_data_array: np.ndarray, 
                           logger: Optional[logging.Logger] = None, 
                           n_jobs: int = -1) -> np.ndarray:
        """Parallel computation of mutual information matrix."""
        N = gene_data_array.shape[0]
        mi_matrix = np.zeros((N, N))
        
        if logger:
            logger.info(f"Computing mutual information matrix for {N} genes (parallel version)")
        
        # Prepare gene pairs
        gene_pairs = []
        for i in range(N):
            for j in range(i+1, N):
                gene_pairs.append((i, j, gene_data_array[i], gene_data_array[j]))
        
        total_pairs = len(gene_pairs)
        if logger:
            logger.info(f"Total gene pairs to compute: {total_pairs}")
            start_time = time.time()
        
        # Parallel computation
        results = Parallel(n_jobs=n_jobs)(
            delayed(MutualInformationCalculator._compute_mi_for_pair)(pair)
            for pair in tqdm(gene_pairs, desc="Computing mutual information")
        )
        
        # Fill matrix
        for i, j, mi_val in results:
            mi_matrix[i, j] = mi_val
            mi_matrix[j, i] = mi_val
        
        if logger:
            total_time = time.time() - start_time
            logger.info(f"Mutual information computation completed in {total_time:.1f} seconds")
        
        return mi_matrix 