import yaml
from typing import Dict, Any


class Config:
    """Class to handle configuration loading and management."""
    
    @staticmethod
    def load_config(config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    @staticmethod
    def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
        """Merge two configuration dictionaries."""
        merged = base_config.copy()
        
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        return update_nested_dict(merged, override_config) 