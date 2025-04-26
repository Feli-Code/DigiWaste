from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path

class DataInterface:
    """Standardizes data exchange between different components of the waste treatment system."""

    def __init__(self):
        self.process_data: Dict[str, pd.DataFrame] = {}
        self.economic_data: Dict[str, Dict] = {}
        self.environmental_data: Dict[str, Dict] = {}

    def prepare_composting_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for composting model analysis.

        Args:
            raw_data: Raw input data

        Returns:
            Processed DataFrame ready for composting model
        """
        required_columns = ['Mc', 'P', 'L', 'C', 'CE', 'H', 'LG', 'A', 'In']
        if not all(col in raw_data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {[col for col in required_columns if col not in raw_data.columns]}")

        processed_data = raw_data.copy()
        
        # Calculate total mass and component fractions
        processed_data['total_mass'] = processed_data[required_columns].sum(axis=1)
        for col in required_columns:
            processed_data[f'{col}_fraction'] = processed_data[col] / processed_data['total_mass']
        
        # Add moisture content validation
        if (processed_data['Mc_fraction'] < 0.4).any() or (processed_data['Mc_fraction'] > 0.65).any():
            raise ValueError("Moisture content should be between 40% and 65% for optimal composting")
        
        # Calculate C/N ratio
        c_content = {
            'C': 6, 'CE': 6, 'H': 10, 'LG': 20, 'L': 25
        }
        n_content = {
            'P': 4  # Protein nitrogen content
        }
        
        total_carbon = sum(processed_data[comp] * c_content[comp] for comp in c_content.keys())
        total_nitrogen = sum(processed_data[comp] * n_content[comp] for comp in n_content.keys())
        processed_data['CN_ratio'] = total_carbon / total_nitrogen
        
        if (processed_data['CN_ratio'] < 20).any() or (processed_data['CN_ratio'] > 40).any():
            raise ValueError("C/N ratio should be between 20 and 40 for optimal composting")

        self.process_data['composting'] = processed_data
        return processed_data

    def prepare_tea_data(self, process_type: str, process_results: Dict) -> Dict:
        """Prepare data for TEA analysis.

        Args:
            process_type: Type of treatment process
            process_results: Results from process simulation

        Returns:
            Dictionary formatted for TEA analysis
        """
        # Extract basic process parameters
        input_amount = process_results.get('input_amount', 0)
        process_efficiency = process_results.get('efficiency', 0)
        
        # Calculate detailed cost components
        capital_costs = {
            'equipment': process_results.get('capital_costs', {}).get('equipment', 0),
            'installation': process_results.get('capital_costs', {}).get('installation', 0),
            'engineering': process_results.get('capital_costs', {}).get('engineering', 0),
            'land': process_results.get('capital_costs', {}).get('land', 0)
        }
        
        operating_costs = {
            'labor': process_results.get('operating_costs', {}).get('labor', 0),
            'utilities': process_results.get('operating_costs', {}).get('utilities', 0),
            'maintenance': process_results.get('operating_costs', {}).get('maintenance', 0),
            'consumables': process_results.get('operating_costs', {}).get('consumables', 0)
        }
        
        # Process-specific revenue calculations
        revenue_streams = {}
        if process_type == 'composting':
            compost_yield = process_results.get('yield', {}).get('compost', 0)
            compost_price = process_results.get('market_prices', {}).get('compost', 50)  # Default €50/ton
            revenue_streams['compost'] = compost_yield * compost_price
            revenue_streams['environmental_benefits'] = process_results.get('environmental_benefits', 0)
        
        tea_data = {
            'process_type': process_type,
            'input_parameters': {
                'waste_amount': input_amount,
                'process_efficiency': process_efficiency,
                'product_yield': process_results.get('yield', {})
            },
            'economic_parameters': {
                'capital_costs': capital_costs,
                'operating_costs': operating_costs,
                'revenue_streams': revenue_streams
            },
            'financial_metrics': {
                'payback_period': None,  # To be calculated
                'npv': None,  # To be calculated
                'irr': None  # To be calculated
            }
        }

        self.economic_data[process_type] = tea_data
        return tea_data

    def prepare_brightway_data(self, process_type: str, process_data: Dict) -> Dict:
        """Format data for future Brightway2 LCA integration.

        Args:
            process_type: Type of treatment process
            process_data: Process-specific data

        Returns:
            Dictionary formatted for Brightway2
        """
        # Define impact categories and their units
        impact_categories = {
            'climate_change': 'kg CO2 eq',
            'ozone_depletion': 'kg CFC-11 eq',
            'acidification': 'kg SO2 eq',
            'eutrophication': 'kg PO4 eq',
            'human_toxicity': 'kg 1,4-DB eq',
            'resource_depletion': 'kg Sb eq'
        }

        # Structure following Brightway2 activity data format
        brightway_data = {
            'name': f"{process_type}_treatment",
            'unit': 'metric ton',
            'exchanges': [
                {
                    'input': ('waste_treatment_db', 'mixed_waste'),
                    'amount': process_data.get('input_amount', 0),
                    'type': 'technosphere'
                },
                {
                    'input': ('waste_treatment_db', 'electricity'),
                    'amount': process_data.get('energy_consumption', 0),
                    'type': 'technosphere'
                },
                {
                    'input': ('waste_treatment_db', 'water'),
                    'amount': process_data.get('water_consumption', 0),
                    'type': 'technosphere'
                }
            ],
            'emissions': {
                'air': {
                    'co2': process_data.get('emissions', {}).get('co2', 0),
                    'ch4': process_data.get('emissions', {}).get('ch4', 0),
                    'n2o': process_data.get('emissions', {}).get('n2o', 0),
                    'nh3': process_data.get('emissions', {}).get('nh3', 0)
                },
                'water': {
                    'nitrogen': process_data.get('emissions', {}).get('n_water', 0),
                    'phosphorus': process_data.get('emissions', {}).get('p_water', 0)
                },
                'soil': {
                    'heavy_metals': process_data.get('emissions', {}).get('heavy_metals', 0)
                }
            },
            'impact_assessment': {
                category: {
                    'value': 0.0,  # To be calculated
                    'unit': unit
                } for category, unit in impact_categories.items()
            },
            'parameters': process_data.get('parameters', {}),
            'database': 'waste_treatment_db'
        }

        self.environmental_data[process_type] = brightway_data
        return brightway_data

    def get_optimization_data(self) -> Dict:
        """Prepare combined data for optimization model.

        Returns:
            Dictionary containing economic and environmental parameters
        """
        return {
            'economic_data': self.economic_data,
            'environmental_data': self.environmental_data,
            'process_constraints': {
                process: data.get('constraints', {})
                for process, data in self.process_data.items()
            }
        }
