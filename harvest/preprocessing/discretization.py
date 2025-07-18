import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from typing import Tuple, Optional
import logging


class ExpressionDiscretizer:
    """Discretize continuous gene expression values."""
    
    @staticmethod
    def discretize(adata, layer: Optional[str] = None, n_bins: int = 5, 
                  max_bins: int = 100, logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Discretize continuous expression values."""
        log_msg = lambda msg: logger.info(msg) if logger else print(msg)
        
        # Get expression matrix
        if layer is None:
            X = adata.X
        else:
            X = adata.layers[layer]
        
        log_msg(f"Discretizing gene expression matrix with shape {X.shape}")
        
        # Ensure X is sparse for memory efficiency
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        elif not isinstance(X, sp.csr_matrix):
            X = X.tocsr()
        
        nonzero_cont = X.data
        
        # Initial histogram approximation
        hist_count, hist_edges = np.histogram(
            nonzero_cont, bins=max_bins, density=False)
        hist_centroids = (hist_edges[0:-1] + hist_edges[1:]) / 2
        
        # K-means clustering of bin centers
        log_msg(f"Using K-means to cluster expression values into {n_bins} bins")
        kmeans = KMeans(n_clusters=n_bins, random_state=2021).fit(
            hist_centroids.reshape(-1, 1), sample_weight=hist_count)
        cluster_centers = np.sort(kmeans.cluster_centers_.flatten())
        
        # Create bin edges
        padding = (hist_edges[-1] - hist_edges[0]) / (max_bins * 10)
        bin_edges = np.array(
            [hist_edges[0] - padding] +
            list((cluster_centers[0:-1] + cluster_centers[1:]) / 2) +
            [hist_edges[-1] + padding])
        
        # Digitize values
        nonzero_disc = np.digitize(nonzero_cont, bin_edges).reshape(-1,)
        bin_count = np.unique(nonzero_disc, return_counts=True)[1]
        
        # Store discretized values
        adata.layers['harvest_discretized'] = X.copy()
        adata.layers['harvest_discretized'].data = nonzero_disc
        
        # Store discretization info
        adata.uns['harvest_disc'] = {
            'bin_edges': bin_edges,
            'bin_count': bin_count,
            'hist_edges': hist_edges,
            'hist_count': hist_count
        }
        
        # Convert to DataFrame
        csr_sparse_data = csr_matrix(adata.layers['harvest_discretized'])
        df_disc = pd.DataFrame(csr_sparse_data.toarray())
        df_disc_named = pd.DataFrame(csr_sparse_data.toarray(), columns=adata.var.index)
        
        log_msg("Discretization completed")
        return df_disc, df_disc_named 