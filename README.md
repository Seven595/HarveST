# HarveST: Heterogeneous Graph Learning Framework for Revealing Spatial Transcriptomics Patterns

HarveST is a Python package for spatial transcriptomics data analysis using graph neural networks with attention mechanisms.

## Installation

### Recommended: Conda Environment Installation
```bash
# 1. Clone the repository
git clone https://github.com/Seven595/HarveST.git
cd HarveST

# 2. Create and activate the Conda environment
conda env create -f environment.yml
conda activate harvest-env

# 3. Install the package in editable mode
pip install -e .
```

## Quick Start

```python
from harvest import Harvest

# Initialize HarveST
harvest = Harvest(output_dir="./results")

# Preprocess data
data = harvest.pre_process(
    data_path="/path/to/visium/data",
    n_top_genes=3000
)

# Perform clustering
results = harvest.cluster(
    n_clusters=7,
    plot_results=True
)
```

## Features

- Data Preprocessing: Load and preprocess 10x Visium data
- Graph Neural Networks: Advanced GNN with attention mechanisms
- Multiple Clustering: Various clustering and refinement strategies
- Spatial Analysis: Spatial-aware clustering refinement
- Visualization: Automatic spatial plot generation

## Requirements
- Python ≥ 3.8
- PyTorch == 1.13.1
- scanpy ≥ 1.8.0
- scikit-learn ≥ 1.0.0
- See environment.yml for full list

## Documentation
Please see example/example_usage.py

## License
This project is licensed under the MIT License. 
