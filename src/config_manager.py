from typing import Dict, Any, Optional
import json
import os
from pathlib import Path

class ConfigManager:
    """Manages configuration settings for waste treatment system components."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or 'config.json'
        self.config: Dict[str, Any] = self._load_default_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings."""
        return {
            'composting': {
                'model_parameters': {
                    'temperature_range': [20, 70],
                    'moisture_range': [40, 65],
                    'cn_ratio_range': [25, 35]
                },
                'tea_parameters': {
                    'price_N': 1.5,
                    'price_P': 2.0,
                    'price_K': 1.0,
                    'compost_per_ton': 0.6
                }
            },
            'lca': {
                'impact_categories': [
                    'Global Warming',
                    'Human Health',
                    'Ecosystem Quality',
                    'Resource Use'
                ],
                'functional_unit': '1 metric ton',
                'system_boundaries': ['gate-to-gate'],
                'brightway_settings': {
                    'project_name': 'waste_treatment_lca',
                    'database_name': 'waste_treatment_db'
                }
            },
            'optimization': {
                'objectives': ['npv', 'environmental_impact'],
                'constraints': {
                    'min_treatment_capacity': 1000,
                    'max_treatment_capacity': 100000
                },
                'weights': {
                    'economic': 0.5,
                    'environmental': 0.5
                }
            }
        }

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file if exists, otherwise return default."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        return self.config

    def save_config(self) -> None:
        """Save current configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def get_composting_config(self) -> Dict[str, Any]:
        """Get composting-specific configuration."""
        return self.config.get('composting', {})

    def get_lca_config(self) -> Dict[str, Any]:
        """Get LCA-specific configuration."""
        return self.config.get('lca', {})

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization-specific configuration."""
        return self.config.get('optimization', {})

    def update_config(self, section: str, parameters: Dict[str, Any]) -> None:
        """Update configuration parameters for a specific section.

        Args:
            section: Configuration section to update
            parameters: New parameters to set
        """
        if section in self.config:
            self.config[section].update(parameters)
        else:
            self.config[section] = parameters
