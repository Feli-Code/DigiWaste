from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from ..data_handling.data_processor import DataProcessor
from ..treatment.composting import CompostingModel
from ...economic_analysis import CompostingTEA, AnaerobicDigestionTEA, IncinerationTEA
from ..treatment.incineration import IncinerationModel

class IntegrationManager:
    """Manages integration between different components of the waste treatment system.

    This class handles the integration of various waste treatment processes, including:
    - Composting process modeling and TEA analysis
    - Life Cycle Assessment (LCA) data preparation
    - Multi-objective optimization parameter management

    The manager maintains process results, LCA results, and TEA results in separate
    dictionaries for each treatment technology. It provides methods to integrate
    these results and prepare them for optimization and analysis.

    Attributes:
        data_processor (DataProcessor): Processor for handling input/output data
        process_results (Dict[str, Dict]): Results from treatment processes
        lca_results (Dict[str, Dict]): Results from LCA calculations
        tea_results (Dict[str, Dict]): Results from TEA calculations
    """

    def __init__(self, data_processor: Optional[DataProcessor] = None):
        self.data_processor = data_processor or DataProcessor()
        self.process_results: Dict[str, Dict] = {}
        self.lca_results: Dict[str, Dict] = {}
        self.tea_results: Dict[str, Dict] = {}

    def integrate_composting_results(self, composting_model: CompostingModel, tea_model: CompostingTEA) -> Dict:
        """Integrate composting model results with TEA analysis and prepare for LCA.

        Args:
            composting_model: Instance of CompostingModel
            tea_model: Instance of CompostingTEA

        Returns:
            Dictionary containing integrated results for TEA and LCA
        """
    def integrate_incineration_results(self, incineration_model: IncinerationModel, tea_model: IncinerationTEA) -> Dict:
        """Integrate incineration model results with TEA analysis and prepare for LCA.

        Args:
            incineration_model: Instance of IncinerationModel
            tea_model: Instance of IncinerationTEA

        Returns:
            Dictionary containing integrated results for TEA and LCA
        """
        process_summary = incineration_model.get_process_summary()
        tea_metrics = tea_model.calculate_metrics()

        integrated_results = {
            'process_metrics': {
                'mass_reduction': 1 - (process_summary['mass_balance']['ash_residue'] 
                                      / process_summary['mass_balance']['input_mass']),
                'energy_efficiency': process_summary['process_efficiency'],
                'emission_factors': process_summary['emissions']
            },
            'economic_metrics': tea_metrics,
            'process_results': {
                'input_amount': process_summary['mass_balance']['input_mass'],
                'efficiency': process_summary['process_efficiency'],
                'yield': {
                    'energy': process_summary['energy_recovery']['recoverable_energy'],
                    'ash': process_summary['mass_balance']['ash_residue']
                },
                'emissions': process_summary['emissions'],
                'energy_consumption': 0  # Placeholder for actual energy consumption data
            }
        }

        self.process_results['incineration'] = integrated_results
        return integrated_results
        """Integrate composting model results with TEA analysis and prepare for LCA.

        Args:
            composting_model: Instance of CompostingModel
            tea_model: Instance of CompostingTEA

        Returns:
            Dictionary containing integrated results for TEA and LCA
        """
        # Extract key metrics from composting model
        composting_metrics = {
            'temperature_profile': composting_model.df_combined.get('T', []),
            'moisture_content': composting_model.df_combined['Mc'] / composting_model.TM_0,
            'organic_matter_degradation': 1 - (composting_model.df_combined[['C', 'CE', 'H', 'LG', 'L']].sum(axis=1) /
                                              composting_model.df_combined[['C', 'CE', 'H', 'LG', 'L']].iloc[0].sum()),
            'cn_ratio': composting_model.calculate_CN_ratio(
                composting_model.df_combined['C'].iloc[-1],
                composting_model.df_combined['P'].iloc[-1],
                composting_model.df_combined['L'].iloc[-1],
                composting_model.df_combined['H'].iloc[-1],
                composting_model.df_combined['CE'].iloc[-1],
                composting_model.df_combined['LG'].iloc[-1]
            ),
            'stability_index': 1 - (composting_model.df_combined[['C', 'CE', 'H', 'LG', 'L']].iloc[-1].sum() /
                                   composting_model.df_combined[['C', 'CE', 'H', 'LG', 'L']].iloc[0].sum())
        }

        # Calculate process efficiency and quality parameters
        process_efficiency = min(1.0, max(0.0,
            0.4 * (1 - composting_metrics['organic_matter_degradation']) +
            0.3 * (1 - abs(composting_metrics['moisture_content'] - 0.55) / 0.55) +
            0.3 * (1 - abs(composting_metrics['cn_ratio'] - 30) / 30)
        ))

        # Get cumulative emissions and process metrics from model
        emissions = composting_model.cumulative_emissions
        compost_production = composting_model.compost_production
        
        # Update compost quality parameters
        compost_production['quality_parameters'] = {
            'moisture_content': composting_metrics['moisture_content'],
            'organic_matter': 1 - composting_metrics['organic_matter_degradation'],
            'cn_ratio': composting_metrics['cn_ratio']
        }

        # Prepare process results for TEA and LCA
        process_results = {
            'input_amount': composting_model.TM_0,
            'efficiency': process_efficiency,
            'yield': {
                'compost': compost_production['mass'],
                'quality': compost_production.get('quality_parameters', {})
            },
            'emissions': emissions,
            'energy_consumption': composting_model.df_combined.get('energy_consumption', 0),
            'water_consumption': composting_model.df_combined.get('water_consumption', 0)
        }

        # Calculate TEA metrics
        tea_metrics = tea_model.calculate_metrics()

        integrated_results = {
            'process_metrics': composting_metrics,
            'economic_metrics': tea_metrics,
            'process_results': process_results
        }

        self.process_results['composting'] = integrated_results
        return integrated_results

    def integrate_anaerobic_digestion_results(self, ad_model, tea_model) -> Dict:
        """Integrate anaerobic digestion results with TEA analysis and prepare for LCA.

        Args:
            ad_model: Instance of AnaerobicDigestionModel
            tea_model: Instance of AnaerobicDigestionTEA

        Returns:
            Dictionary containing integrated results for TEA and LCA
        """
        process_metrics = {
            'biogas_production': ad_model.biogas_yield,
            'digestate_quality': ad_model.digestate_quality,
            'methane_content': ad_model.methane_percentage
        }

        process_efficiency = min(1.0, max(0.0,
            0.6 * (ad_model.biogas_yield / ad_model.input_mass) +
            0.4 * (ad_model.methane_percentage / 100)
        ))

        tea_metrics = tea_model.calculate_metrics()

        integrated_results = {
            'process_metrics': process_metrics,
            'economic_metrics': tea_metrics,
            'process_results': {
                'input_amount': ad_model.input_mass,
                'efficiency': process_efficiency,
                'yield': {
                    'biogas': ad_model.biogas_yield,
                    'digestate': ad_model.digestate_mass
                },
                'emissions': ad_model.emissions,
                'energy_consumption': ad_model.energy_balance
            }
        }

        self.process_results['anaerobic_digestion'] = integrated_results
        return integrated_results

    def prepare_lca_data(self, process_type: str, process_data: Dict) -> Dict:
        """Prepare data for LCA analysis and optimization, with future Brightway compatibility.

        Args:
            process_type: Type of treatment process
            process_data: Process-specific data

        Returns:
            Dictionary formatted for LCA analysis and optimization
        """
        # Calculate environmental impact factors
        emissions = process_data.get('emissions', {})
        gwp_factors = {
            'CO2': 1,      # GWP factor for CO2
            'CH4': 28,     # GWP factor for CH4 (100-year)
            'N2O': 265,    # GWP factor for N2O (100-year)
            'NH3': 0       # NH3 doesn't have direct GWP
        }

        # Calculate total GWP
        # Calculate comprehensive environmental impacts
        impact_factors = {
            'climate_change': {
                'CO2': 1,
                'CH4': 28,
                'N2O': 265
            },
            'ozone_depletion': {'CFC-11': 1},
            'acidification': {'SO2': 1, 'NH3': 1.6},
            'eutrophication': {'NOx': 0.13, 'PO4': 1},
            'human_toxicity': {'heavy_metals': 10}
        }

        impact_values = {
            category: sum(
                emissions.get(gas, 0) * factor
                for gas, factor in factors.items()
            )
            for category, factors in impact_factors.items()
        }

        # Structure data for future Brightway integration
        lca_data = {
            'inputs': {
                'waste_amount': process_data.get('input_amount', 0),
                'energy_consumption': process_data.get('energy_consumption', 0),
                'water_consumption': process_data.get('water_consumption', 0),
                'chemical_agents': process_data.get('chemical_agents', 0),
                'labor_hours': process_data.get('labor_hours', 0)
            },
            'outputs': {
                'products': process_data.get('yield', {}),
                'emissions': emissions,
                'process_metrics': {
                    'temperature_profile': process_data.get('temperature_profile', []),
                    'moisture_content': process_data.get('moisture_content', []),
                    'organic_matter_degradation': process_data.get('organic_matter_degradation', 0),
                    'stability_index': process_data.get('stability_index', 0)
                }
            },
            'impact_assessment': {
                'human_health': {
                    'climate_change': impact_values.get('climate_change', 0.0),
                    'ozone_formation': process_data.get('impact_ozone_formation', 0.0),
                    'ozone_depletion': process_data.get('impact_ozone_depletion', 0.0),
                    'carcinogenic_toxicity': process_data.get('impact_carcinogenic', 0.0),
                    'non_carcinogenic_toxicity': process_data.get('impact_non_carcinogenic', 0.0)
                },
                'ecosystem': {
                    'acidification': impact_values.get('acidification', 0.0),
                'ozone_depletion': impact_values.get('ozone_depletion', 0.0),
                'eutrophication': impact_values.get('eutrophication', 0.0),
                'human_toxicity': impact_values.get('human_toxicity', 0.0),
                    'terrestrial_ecotoxicity': process_data.get('impact_terrestrial_eco', 0.0),
                    'freshwater_eutrophication': process_data.get('impact_eutrophication', 0.0),
                    'freshwater_ecotoxicity': process_data.get('impact_freshwater_eco', 0.0),
                    'marine_ecotoxicity': process_data.get('impact_marine_eco', 0.0)
                },
                'resources': {
                    'fossil_scarcity': process_data.get('impact_fossil_scarcity', 0.0),
                    'mineral_scarcity': process_data.get('impact_mineral_scarcity', 0.0)
                }
            },
            'metadata': {
                'process_type': process_type,
                'functional_unit': '1 metric ton of waste',
                'system_boundaries': ['gate-to-gate'],
                'impact_categories': [
                    'climate_change',
                    'human_toxicity',
                    'ozone_depletion',
                    'acidification',
                    'eutrophication',
                    'resource_depletion'
                ],
                'temporal_scope': {
                    'duration': 1,
                    'unit': 'year',
                    'reference_year': 2023
                },
                'geographical_scope': {
                    'location': 'facility-level',
                    'coordinates': None,
                    'brightway_region': 'GLO'
                },
                'data_sources': ['ecoinvent', 'local_data'],
                'brightway_compatibility': {
                    'database': 'waste_treatment_db',
                    'version': '3.8',
                    'migration_notes': 'Requires biosphere3 updates'
                }
            }
        }

        self.lca_results[process_type] = lca_data
        return lca_data

    def get_optimization_parameters(self) -> Dict:
        """Aggregate optimization parameters from all treatment processes for multi-objective optimization.

        Returns:
            Dictionary containing combined parameters for:
            - Mass balance
            - Energy consumption
            - Economic metrics
            - Environmental impacts
        """
        params = {
            'inputs': {},
            'outputs': {},
            'constraints': {},
            'objectives': {}
        }

        # Aggregate parameters from all processes
        for process_name, results in self.process_results.items():
            prefix = f"{process_name}_"
            process_data = results['process_results']

            # Mass balance parameters
            params['inputs'][f"{prefix}input_mass"] = process_data['input_amount']
            params['outputs'][f"{prefix}output_mass"] = process_data['yield'].get('compost', 0) \
                or process_data['yield'].get('energy', 0) \
                or process_data['yield'].get('biogas', 0)

            # Efficiency parameters
            params['constraints'][f"{prefix}min_efficiency"] = 0.7
            params['objectives'][f"{prefix}efficiency"] = process_data['efficiency']

            # Economic parameters
            economic = results['economic_metrics']
            params['objectives'][f"{prefix}npv"] = economic.get('net_present_value', 0)
            params['constraints'][f"{prefix}min_irr"] = economic.get('internal_rate_of_return', 0)

            # Environmental parameters
            params['objectives'][f"{prefix}co2_equiv"] = process_data['emissions'].get('co2', 0) \
                + process_data['emissions'].get('ch4', 0) * 25

        # Cross-process constraints
        params['constraints']['max_total_energy'] = sum(
            p['process_results']['energy_consumption'] 
            for p in self.process_results.values()
        )

        params['constraints']['max_total_water'] = sum(
            p['process_results'].get('water_consumption', 0)
            for p in self.process_results.values()
        )

        return params
        """Prepare parameters for the optimization model with enhanced resource tracking."""
        optimization_params = {
            'economic_parameters': {
                process: {
                    'capex': results.get('economic_metrics', {}).get('capex', 0),
                    'opex': results.get('economic_metrics', {}).get('opex', 0),
                    'revenue': results.get('economic_metrics', {}).get('revenue', 0),
                    'roi': results.get('economic_metrics', {}).get('roi', 0),
                    'payback_period': results.get('economic_metrics', {}).get('payback_period', 0),
                    'unit_processing_cost': results.get('economic_metrics', {}).get('unit_processing_cost', 0),
                    'product_value': results.get('economic_metrics', {}).get('product_value', 0),
                    'process_efficiency': results.get('process_results', {}).get('efficiency', 0),
                    'product_quality': results.get('process_results', {}).get('yield', {}).get('quality', {}),
                    'resource_utilization': {
                        'energy_efficiency': results.get('process_results', {}).get('energy_efficiency', 0),
                        'water_efficiency': results.get('process_results', {}).get('water_efficiency', 0),
                        'labor_productivity': results.get('process_results', {}).get('labor_productivity', 0)
                    },
                    'market_factors': {
                        'product_demand': results.get('market_analysis', {}).get('demand_factor', 1.0),
                        'price_volatility': results.get('market_analysis', {}).get('price_volatility', 0.1),
                        'competition_index': results.get('market_analysis', {}).get('competition_index', 0.5)
                    }
                }
                for process, results in self.process_results.items()
            },
            'environmental_parameters': {
                process: {
                    'emissions': results.get('process_results', {}).get('emissions', {}),
                    'resource_consumption': {
                        'energy': results.get('process_results', {}).get('energy_consumption', 0),
                        'water': results.get('process_results', {}).get('water_consumption', 0),
                        'raw_materials': results.get('process_results', {}).get('raw_materials_consumption', {}),
                        'chemicals': results.get('process_results', {}).get('chemical_consumption', {})
                    },
                    'impact_assessment': {
                        'human_health': {
                            'gwp': self._calculate_gwp(results.get('process_results', {}).get('emissions', {})),
                            'ozone_formation': results.get('impact_assessment', {}).get('human_health', {}).get('ozone_formation', 0),
                            'particulate_matter': results.get('impact_assessment', {}).get('human_health', {}).get('particulate_matter', 0),
                            'toxicity_potential': results.get('impact_assessment', {}).get('human_health', {}).get('toxicity_potential', 0),
                            'ozone_depletion': results.get('impact_assessment', {}).get('human_health', {}).get('ozone_depletion', 0),
                            'carcinogenic_toxicity': results.get('impact_assessment', {}).get('human_health', {}).get('carcinogenic_toxicity', 0),
                            'non_carcinogenic_toxicity': results.get('impact_assessment', {}).get('human_health', {}).get('non_carcinogenic_toxicity', 0)
                        },  # Properly close human_health dict
                        'ecosystem': {
                            'acidification': self._calculate_acidification(results.get('process_results', {}).get('emissions', {})),
                            'eutrophication': self._calculate_eutrophication(results.get('process_results', {}).get('emissions', {})),
                            'terrestrial_ecotoxicity': results.get('impact_assessment', {}).get('ecosystem', {}).get('terrestrial_ecotoxicity', 0),
                            'freshwater_ecotoxicity': results.get('impact_assessment', {}).get('ecosystem', {}).get('freshwater_ecotoxicity', 0),
                            'marine_ecotoxicity': results.get('impact_assessment', {}).get('ecosystem', {}).get('marine_ecotoxicity', 0)
                        },
                        'resources': {
                            'fossil_scarcity': results.get('impact_assessment', {}).get('resources', {}).get('fossil_scarcity', 0),
                            'mineral_scarcity': results.get('impact_assessment', {}).get('resources', {}).get('mineral_scarcity', 0),
                            'resource_depletion': self._calculate_resource_depletion(results)
                        }
                    },
                    'process_efficiency': results.get('outputs', {}).get('process_metrics', {}).get('stability_index', 0)
                }
                for process, results in self.lca_results.items()
            },
            'operational_parameters': {
                process: {
                    'temperature_profile': results.get('outputs', {}).get('process_metrics', {}).get('temperature_profile', []),
                    'moisture_content': results.get('outputs', {}).get('process_metrics', {}).get('moisture_content', []),
                    'organic_matter_degradation': results.get('outputs', {}).get('process_metrics', {}).get('organic_matter_degradation', 0),
                    'cn_ratio': results.get('outputs', {}).get('process_metrics', {}).get('cn_ratio', 0),
                    'stability_index': results.get('outputs', {}).get('process_metrics', {}).get('stability_index', 0),
                    'process_duration': results.get('outputs', {}).get('process_metrics', {}).get('process_duration', 0)
                }
                for process, results in self.lca_results.items()
            }
        }       
    
        return optimization_params

    def _calculate_gwp(self, results: Dict) -> float:
        """Calculate Global Warming Potential from emissions data."""
        emissions = results.get('outputs', {}).get('emissions', {})
        # CO2 equivalent factors (IPCC AR5 100-year)
        co2_eq = {
            'co2': 1,
            'ch4': 28,  # Methane
            'n2o': 265,  # Nitrous oxide
            'nh3': 0,    # Ammonia (indirect through N cycle)
            'voc': 3.4   # Volatile organic compounds (averaged)
        }
        # Calculate direct GWP
        direct_gwp = sum(emissions.get(gas, 0) * factor for gas, factor in co2_eq.items())
        
        # Add indirect effects (e.g. N deposition from NH3)
        nh3_n2o_conversion = 0.01  # 1% of NH3-N converts to N2O-N
        indirect_n2o = emissions.get('nh3', 0) * nh3_n2o_conversion * co2_eq['n2o']
        
        return direct_gwp + indirect_n2o

    def _calculate_acidification(self, results: Dict) -> float:
        """Calculate Acidification Potential from emissions data."""
        emissions = results.get('outputs', {}).get('emissions', {})
        # SO2 equivalent factors (based on latest characterization factors)
        so2_eq = {
            'nh3': 1.88,  # NH3 to SO2 equivalent
            'nox': 0.7,   # NOx to SO2 equivalent
            'so2': 1.0,   # SO2 baseline
            'hcl': 0.88,  # Hydrochloric acid
            'h2s': 1.88   # Hydrogen sulfide
        }
        # Calculate total acidification potential
        ap = sum(emissions.get(gas, 0) * factor for gas, factor in so2_eq.items())
        
        # Consider pH buffering capacity if available
        buffer_capacity = results.get('soil_properties', {}).get('buffer_capacity', 1.0)
        return ap * buffer_capacity

    def _calculate_eutrophication(self, results: Dict) -> float:
        """Calculate Eutrophication Potential from emissions data."""
        emissions = results.get('outputs', {}).get('emissions', {})
        # PO4 equivalent factors (updated with latest characterization factors)
        po4_eq = {
            'nh3': 0.35,  # NH3 to PO4 equivalent
            'nox': 0.13,  # NOx to PO4 equivalent
            'n': 0.42,    # Total N
            'p': 3.06,    # Total P
            'cod': 0.022  # Chemical Oxygen Demand
        }
        # Calculate aquatic and terrestrial eutrophication
        aquatic_ep = sum(emissions.get(gas, 0) * factor 
                        for gas, factor in po4_eq.items())
        
        # Consider local sensitivity factors
        water_body_sensitivity = results.get('environmental_conditions', {}).get('water_sensitivity', 1.0)
        return aquatic_ep * water_body_sensitivity

    def _calculate_resource_depletion(self, results: Dict) -> float:
        """Calculate Resource Depletion score from input consumption."""
        inputs = results.get('inputs', {})
        # Resource depletion factors (normalized)
        factors = {
            'energy_consumption': 0.4,
            'water_consumption': 0.3,
            'chemical_agents': 0.2,
            'labor_hours': 0.1
        }
        return sum(inputs.get(resource, 0) * factor for resource, factor in factors.items())

    def export_results(self, format: str = 'csv') -> Dict[str, str]:
        """Export integrated results in specified format.

        Args:
            format: Output format ('csv' or 'json')

        Returns:
            Dictionary with paths to exported files
        """
        export_paths = {}
        
        # Export process results
        process_df = pd.DataFrame(self.process_results)
        if format == 'csv':
            process_df.to_csv('process_results.csv')
            export_paths['process_results'] = 'process_results.csv'
        else:
            process_df.to_json('process_results.json')
            export_paths['process_results'] = 'process_results.json'

        return export_paths
