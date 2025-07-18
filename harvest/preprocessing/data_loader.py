import os
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Tuple, Dict, Any, Optional
import logging

from .spatial_distance import SpatialDistanceCalculator
from .mutual_information import MutualInformationCalculator
from .discretization import ExpressionDiscretizer


class DataPreprocessor:
    """Main class for data preprocessing operations."""
    
    def __init__(self, output_dir: str, logger: Optional[logging.Logger] = None, random_seed: int = 2023):
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        self.random_seed = random_seed
        
    def load_visium(self, data_path: str, count_file: str = "filtered_feature_bc_matrix.h5"):
        """Load 10x Visium spatial transcriptomics data."""
        self.logger.info(f"Loading Visium data from: {data_path}")
        
        try:
            adata = sc.read_visium(data_path, count_file=count_file, load_images=True)
            adata.var_names_make_unique()
            self.logger.info(f"Successfully loaded Visium data with shape {adata.shape}")
            return adata
        except Exception as e:
            self.logger.error(f"Error loading Visium data: {str(e)}")
            raise
    
    def preprocess_data(self, adata, n_top_genes: int = 3000):
        """Preprocess AnnData object."""
        self.logger.info("Preprocessing data...")
        
        # Highly variable genes and normalization
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Subset to highly variable genes
        adata_vars = adata[:, adata.var['highly_variable']]
        self.logger.info(f"Selected {adata_vars.shape[1]} highly variable genes")
        
        return adata_vars
    
    def extract_features(self, adata) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """Extract cell and gene features from AnnData object."""
        self.logger.info(f"Extracting features: {adata.shape[0]} cells, {adata.shape[1]} genes")
        
        # Create ID mappings
        cell_mapping = dict(zip(adata.obs.index, range(len(adata.obs.index))))
        gene_mapping = dict(zip(adata.var.index, range(len(adata.var.index))))
        
        # Extract feature matrices
        feat_cell = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
        feat_gene = np.transpose(feat_cell)
        
        return feat_cell, feat_gene, cell_mapping, gene_mapping
    
    def generate_mutual_information(self, gene_features: np.ndarray, 
                                  parallel: bool = True, n_jobs: int = -1) -> np.ndarray:
        """Generate gene-gene mutual information matrix."""
        if parallel:
            return MutualInformationCalculator.generate_mi_parallel(
                gene_features, logger=self.logger, n_jobs=n_jobs)
        else:
            return MutualInformationCalculator.generate_mi(
                gene_features, logger=self.logger)
    
    def generate_expression_matrix(self, adata, n_bins: int = 5) -> np.ndarray:
        """Generate discretized cell-gene expression matrix."""
        expr_matrix, _ = ExpressionDiscretizer.discretize(
            adata, n_bins=n_bins, logger=self.logger)
        return expr_matrix.values
    
    def generate_spatial_adjacency(self, adata, p: float = 1.0, radius: float = 150,
                                 start: float = 1, end: float = 300) -> np.ndarray:
        """Generate cell-cell spatial adjacency matrix."""
        adj_df = SpatialDistanceCalculator.cal_cc_weight(
            adata, p=p, radius=radius, start=start, end=end, logger=self.logger)
        return adj_df.values 