"""
Spatial Gene Discovery Module for HarveST
Discovers domain-specific spatial variable genes using random walk with restart (RWR)
"""

import numpy as np
import pandas as pd
import scanpy as sc
import os
import random
import scipy.sparse as sp
from tqdm import tqdm
from pathlib import Path
import warnings
from typing import Dict, List, Optional, Union, Any
import logging

warnings.filterwarnings("ignore")


class SpatialGeneDiscovery:
    """
    A class for discovering domain-specific spatial variable genes (SVGs) using 
    random walk with restart (RWR) on a heterogeneous network.
    
    This class integrates with the HarveST framework to provide spatial gene discovery
    capabilities for spatial transcriptomics data analysis.
    """
    
    def __init__(self, data_dir: Union[str, Path], output_dir: Optional[Union[str, Path]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the SpatialGeneDiscovery object.
        
        Args:
            data_dir: Directory containing the data files
            output_dir: Directory to save output files (default: creates 'svg_results' in data_dir)
            logger: Logger instance for logging messages
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "svg_results"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up logging
        if logger is None:
            from .utils import setup_logger
            self.logger = setup_logger(str(self.output_dir), "spatial_gene_discovery")
        else:
            self.logger = logger
        
        # Initialize data objects
        self.adata = None
        self.adj_matrix = None
        self.num_cells = None
        self.num_genes = None
        self.gene_name_mapping = None
        self.domains = None
        
    def setup_renv(self, r_home_path: str):
        """Set up R environment path if needed."""
        os.environ['R_HOME'] = r_home_path
        self.logger.info(f"R environment set to: {r_home_path}")
        
    def load_visium_data(self, section_id: str, count_file: str = 'filtered_feature_bc_matrix.h5', 
                         annotation_file: Optional[str] = None, annotation_column: str = 'ground_truth', 
                         domain_column: str = 'Ground Truth', n_top_genes: int = 3000):
        """
        Load and preprocess Visium spatial transcriptomics data.
        
        Args:
            section_id: ID of the section
            count_file: Name of the count file
            annotation_file: Path to the annotation file (optional)
            annotation_column: Column in annotation file containing domain labels
            domain_column: Name to use for domain information in adata.obs
            n_top_genes: Number of top highly variable genes to select
        """
        file_fold = self.data_dir / section_id
        
        # Check if count_file is a full path or just a filename
        if os.path.isabs(count_file):
            count_file_path = count_file
        else:
            count_file_path = file_fold / count_file
    
        self.logger.info(f"Loading Visium data from: {file_fold}")
        self.logger.info(f"Count file: {count_file_path}")
        
        self.adata = sc.read_visium(file_fold, count_file=count_file_path, load_images=True)
        self.adata.var_names_make_unique()
        
        # Normalize and log-transform
        sc.pp.highly_variable_genes(self.adata, flavor="seurat_v3", n_top_genes=n_top_genes)
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        
        # Load annotation if provided
        if annotation_file:
            ann_path = self.data_dir / annotation_file if not os.path.isabs(annotation_file) else annotation_file
            self.logger.info(f"Loading annotations from: {ann_path}")
            ann_df = pd.read_csv(ann_path, index_col=1)
            self.adata.obs[domain_column] = ann_df.loc[self.adata.obs_names, annotation_column]
            
        # Extract domains
        self.domains = self.adata.obs[domain_column].unique().tolist()
        self.logger.info(f"Loaded data with {len(self.domains)} domains: {self.domains}")
        self.num_cells = self.adata.shape[0]
        self.num_genes = n_top_genes

    def load_h5ad_data(self, h5ad_file: Union[str, Path], annotation_file: Optional[Union[str, Path]] = None, 
                      annotation_column: Optional[str] = None, domain_column: str = 'Ground Truth', 
                      n_top_genes: int = 3000):
        """
        Load and preprocess data from an H5AD file.
        
        Args:
            h5ad_file: Path to the H5AD file
            annotation_file: Path to the annotation file (optional)
            annotation_column: Column in annotation file containing domain labels
            domain_column: Name to use for domain information in adata.obs
            n_top_genes: Number of top highly variable genes to select
        """
        self.logger.info(f"Loading H5AD data from: {h5ad_file}")
        self.adata = sc.read_h5ad(h5ad_file)
        self.adata.var_names_make_unique()
        
        # Normalize and log-transform
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(self.adata, flavor="seurat_v3", n_top_genes=n_top_genes)
        
        # Load annotation if provided
        if annotation_file:
            try:
                self.logger.info(f"Loading annotations from: {annotation_file}")
                ann_df = pd.read_csv(annotation_file, index_col="ID")
                
                columns_to_add = [domain_column] if domain_column in ann_df.columns else ann_df.columns
                
                # Check index matching
                common_indices = set(ann_df.index).intersection(set(self.adata.obs_names))
                if len(common_indices) == 0:
                    self.logger.warning("No common indices found between annotation file and data")
                    self.logger.info(f"Annotation indices (first 5): {list(ann_df.index)[:5]}")
                    self.logger.info(f"Data observation names (first 5): {list(self.adata.obs_names)[:5]}")
                
                # Add annotation columns
                for col in columns_to_add:
                    if col in ann_df.columns:
                        try:
                            self.adata.obs[col] = ann_df.reindex(index=self.adata.obs_names)[col].values
                            self.logger.info(f"Successfully added column '{col}' to adata.obs")
                        except Exception as e:
                            self.logger.error(f"Error adding column '{col}': {str(e)}")
                            if len(ann_df) == self.adata.shape[0]:
                                self.logger.info(f"Assigning column '{col}' by position (row count match)")
                                self.adata.obs[col] = ann_df[col].values
            except Exception as e:
                self.logger.error(f"Error loading annotation file: {str(e)}")
        
        # Extract domains
        if domain_column in self.adata.obs.columns:
            self.domains = self.adata.obs[domain_column].unique().tolist()
            self.logger.info(f"Loaded data with {len(self.domains)} domains: {self.domains}")
        else:
            self.logger.warning(f"{domain_column} not found in adata.obs. No domain information available.")
            self.domains = []
        
        self.num_cells = self.adata.shape[0]
        self.num_genes = n_top_genes
        self.logger.info(f"Loaded dataset with {self.num_cells} cells and selected {self.num_genes} variable genes")
        
    def load_network_data(self, gene_gene_file: Union[str, Path], cell_cell_file: Union[str, Path], 
                         cell_gene_file: Union[str, Path], gene_name_file: Optional[Union[str, Path]] = None):
        """
        Load network data files for cell-cell, gene-gene, and cell-gene relationships.
        
        Args:
            gene_gene_file: Path to gene-gene relationship file
            cell_cell_file: Path to cell-cell relationship file
            cell_gene_file: Path to cell-gene relationship file
            gene_name_file: Path to gene name mapping file (optional)
        """
        def resolve_path(file_path):
            if os.path.isabs(file_path):
                return file_path
            return self.data_dir / file_path
        
        # Load relationship matrices
        gg_path = resolve_path(gene_gene_file)
        cc_path = resolve_path(cell_cell_file)
        cg_path = resolve_path(cell_gene_file)
        
        self.logger.info(f"Loading gene-gene relationships from: {gg_path}")
        self.logger.info(f"Loading cell-cell relationships from: {cc_path}")
        self.logger.info(f"Loading cell-gene relationships from: {cg_path}")
        
        gg = pd.read_csv(gg_path, index_col=0)
        cc = pd.read_csv(cc_path, index_col=0)
        cg = pd.read_csv(cg_path, index_col=0)
        
        # Load gene name mapping if provided
        if gene_name_file:
            gn_path = resolve_path(gene_name_file)
            self.logger.info(f"Loading gene name mapping from: {gn_path}")
            gene_name = np.load(gn_path, allow_pickle=True)
            self.gene_name_mapping = gene_name.item()
        
        # Normalize matrices
        cg_norm = self._normalize_matrix(cg.values)
        gg_norm = self._normalize_matrix(gg.values)
        cc_norm = self._normalize_matrix(cc.values)
        
        # Create integrated adjacency matrix
        self.adj_matrix = np.zeros((self.num_cells + self.num_genes, 
                                    self.num_cells + self.num_genes))
        
        # Fill in relationships
        self.adj_matrix[:self.num_cells, :self.num_cells] = cc_norm
        self.adj_matrix[:self.num_cells, self.num_cells:] = cg_norm
        self.adj_matrix[self.num_cells:, :self.num_cells] = cg_norm.T
        self.adj_matrix[self.num_cells:, self.num_cells:] = gg_norm
        
        self.logger.info(f"Created integrated network with shape {self.adj_matrix.shape}")
        
    def find_domain_specific_genes(self, domain_column: str = 'Ground Truth', restart_prob: float = 0.5, 
                                  p_value_threshold: float = 0.01, num_randomizations: int = 500, 
                                  domains: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Find domain-specific genes for all or specified domains.
        
        Args:
            domain_column: Column in adata.obs containing domain information
            restart_prob: Restart probability for random walk with restart
            p_value_threshold: Threshold for significant p-values
            num_randomizations: Number of random walks for null distribution
            domains: List of specific domains to analyze (if None, analyze all domains)
            
        Returns:
            results: Dictionary mapping domain names to DataFrames of significant genes
        """
        if domains is None:
            domains = self.domains
            
        results = {}
        
        # Create spot to index mapping
        spot_to_index = {spot: idx for idx, spot in enumerate(self.adata.obs_names)}
        
        for domain in tqdm(domains, desc="Processing domains"):
            self.logger.info(f"Analyzing domain: {domain}")
            
            # Get spots in this domain
            domain_spots = self.adata.obs[self.adata.obs[domain_column] == domain].index
            
            if len(domain_spots) == 0:
                self.logger.warning(f"No spots found for domain {domain}, skipping...")
                continue
                
            # Get indices of spots in this domain
            seed_indices = [spot_to_index[spot] for spot in domain_spots]
            
            # Run RWR
            prob_vectors = self._random_walk_with_restart(
                self.adj_matrix, restart_prob, seed_indices
            )
            gene_scores = prob_vectors[-1][self.num_cells:]
            
            # Generate null distribution
            null_distribution = self._generate_null_distribution(
                self.adj_matrix, restart_prob, self.num_cells, self.num_genes, 
                seed_indices, num_randomizations
            )
            
            # Calculate statistics
            p_values = self._calculate_p_values(gene_scores, null_distribution)
            scaled_gene_scores = self._scale_scores_to_01(gene_scores)
            fold_changes = self._calculate_fold_change(gene_scores, null_distribution)
            
            # Filter significant genes
            significant_indices, significant_scores = self._filter_significant_genes(
                gene_scores, p_values, p_value_threshold
            )
            
            if len(significant_indices) == 0:
                self.logger.warning(f"No significant genes found for domain {domain}")
                continue
                
            # Calculate expression-based fold change
            significant_gene_expressions = self.adata[:, significant_indices].X
            
            # Calculate mean expression for domain vs others
            mean_expression_domain = np.mean(significant_gene_expressions[seed_indices], axis=0)
            
            other_indices_mask = np.ones(self.adata.shape[0], dtype=bool)
            other_indices_mask[seed_indices] = False
            mean_expression_others = np.mean(significant_gene_expressions[other_indices_mask], axis=0)
            
            expression_fold_change = mean_expression_domain / (mean_expression_others + 1e-6)
            expression_fold_change = np.array(expression_fold_change).flatten()
            
            # Create results DataFrame
            index_to_gene_name = {v: k for k, v in self.gene_name_mapping.items()}
            significant_genes_data = {
                'Gene': [index_to_gene_name[idx] for idx in significant_indices],
                'Score': significant_scores,
                'Scaled_Score': scaled_gene_scores[significant_indices],
                'p_value': p_values[significant_indices],
                'Network_Fold_Change': fold_changes[significant_indices],
                'Expression_Fold_Change': expression_fold_change
            }
            
            significant_genes_df = pd.DataFrame(significant_genes_data)
            significant_genes_df = significant_genes_df.sort_values('Score', ascending=False)
            
            # Save results
            safe_domain_name = self._sanitize_filename(domain)
            output_file = self.output_dir / f"{safe_domain_name}_specific_genes.csv"
            
            try:
                significant_genes_df.to_csv(output_file, index=False)
                self.logger.info(f"Saved {len(significant_indices)} significant genes for domain {domain} to {output_file}")
            except OSError as e:
                self.logger.error(f"Error saving results for domain {domain}: {str(e)}")
                safe_domain_name = ''.join(c if c.isalnum() else '_' for c in domain)
                output_file = self.output_dir / f"{safe_domain_name}_specific_genes.csv"
                significant_genes_df.to_csv(output_file, index=False)
                self.logger.info(f"Saved with alternative filename: {output_file}")
            
            results[domain] = significant_genes_df
            
        return results

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by replacing problematic characters."""
        return filename.replace('/', '_').replace('\\', '_').replace(' ', '_')

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize a matrix by row sums."""
        row_sums = matrix.sum(axis=1)
        normalized_matrix = matrix / row_sums[:, np.newaxis]
        return np.nan_to_num(normalized_matrix)
    
    def _random_walk_with_restart(self, adj_matrix: np.ndarray, restart_prob: float, 
                                 seed_indices: List[int], max_iter: int = 100, 
                                 tol: float = 1e-6) -> List[np.ndarray]:
        """
        Perform random walk with restart.
        
        Args:
            adj_matrix: Adjacency matrix of the graph
            restart_prob: Probability of restarting the walk
            seed_indices: Indices of seed nodes
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            prob_vectors: List of probability vectors at each iteration
        """
        num_nodes = adj_matrix.shape[0]
        
        # Initialize restart vector
        restart_vector = np.zeros(num_nodes)
        restart_vector[seed_indices] = 1 / len(seed_indices)
        
        # Initialize probability vector
        prob_vector = restart_vector.copy()
        
        # Normalize adjacency matrix
        normalized_adj_matrix = self._normalize_matrix(adj_matrix)
        
        prob_vectors = [prob_vector.copy()]
        
        for iteration in range(max_iter):
            new_prob_vector = (1 - restart_prob) * np.dot(normalized_adj_matrix, prob_vector) + restart_prob * restart_vector
            prob_vectors.append(new_prob_vector.copy())
            
            if np.linalg.norm(new_prob_vector - prob_vector) < tol:
                self.logger.info(f"RWR converged after {iteration+1} iterations")
                break
                
            prob_vector = new_prob_vector
        
        return prob_vectors
    
    def _generate_null_distribution(self, adj_matrix: np.ndarray, restart_prob: float, 
                                   num_cells: int, num_genes: int, seed_indices: List[int], 
                                   num_randomizations: int = 500) -> np.ndarray:
        """
        Generate null distribution by random walks from random seed sets.
        
        Args:
            adj_matrix: Adjacency matrix
            restart_prob: Restart probability
            num_cells: Number of cells
            num_genes: Number of genes
            seed_indices: Indices of seed nodes in the real case
            num_randomizations: Number of random walks
            
        Returns:
            null_scores: Null distribution of gene scores
        """
        null_scores = np.zeros((num_randomizations, num_genes))
        
        for i in tqdm(range(num_randomizations), desc="Generating null distribution"):
            random_seed_indices = np.random.choice(num_cells, size=len(seed_indices), replace=False)
            prob_vectors = self._random_walk_with_restart(adj_matrix, restart_prob, random_seed_indices)
            null_scores[i, :] = prob_vectors[-1][num_cells:]
        
        return null_scores
    
    def _calculate_p_values(self, gene_scores: np.ndarray, null_distribution: np.ndarray) -> np.ndarray:
        """Calculate empirical p-values from null distribution."""
        p_values = (np.sum(null_distribution >= gene_scores, axis=0) + 1) / (null_distribution.shape[0] + 1)
        return p_values
    
    def _scale_scores_to_01(self, gene_scores: np.ndarray) -> np.ndarray:
        """Scale scores to range [0, 1]."""
        min_score = np.min(gene_scores)
        max_score = np.max(gene_scores)
        scaled_scores = (gene_scores - min_score) / (max_score - min_score)
        return scaled_scores
    
    def _calculate_fold_change(self, gene_scores: np.ndarray, null_distribution: np.ndarray) -> np.ndarray:
        """Calculate fold change relative to null distribution mean."""
        null_mean = np.mean(null_distribution, axis=0)
        fold_change = gene_scores / (null_mean + 1e-6)
        return fold_change
    
    def _filter_significant_genes(self, gene_scores: np.ndarray, p_values: np.ndarray, 
                                 p_value_threshold: float = 0.05) -> tuple:
        """
        Filter significant genes based on p-value threshold.
        
        Args:
            gene_scores: Scores for each gene
            p_values: p-values for each gene
            p_value_threshold: Threshold for significance
            
        Returns:
            significant_indices: Indices of significant genes
            significant_genes: Scores of significant genes
        """
        significant_indices = np.where(p_values < p_value_threshold)[0]
        significant_genes = gene_scores[significant_indices]
        return significant_indices, significant_genes