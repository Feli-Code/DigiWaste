class AnaerobicDigestionModel:
    """Anaerobic digestion process model for biowaste treatment

    Handles mass balance calculations, biogas production estimation,
    digestate outputs, and energy recovery calculations.
    """

    def __init__(self, waste_composition: dict):
        """Initialize AD model parameters

        Args:
            waste_composition: Dictionary containing waste characteristics
                Expected keys: Mc (moisture), P (proteins), L (lipids), C (carbohydrates)
        """
        self.waste_composition = waste_composition
        self.results = {
            'mass_balance': {},
            'biogas_production': {},
            'energy_recovery': {},
            'digestate_outputs': {}
        }

    def calculate_mass_balance(self, input_mass: float) -> dict:
        """Calculate mass balance for AD process

        Args:
            input_mass: Total mass of waste input (kg)

        Returns:
            Dictionary with mass balance components
        """
        # Calculate dry matter content
        dry_mass = input_mass * (1 - self.waste_composition['Mc'])
        
        # Biochemical composition calculations
        proteins = dry_mass * self.waste_composition['P']
        lipids = dry_mass * self.waste_composition['L']
        carbohydrates = dry_mass * self.waste_composition['C']

        # Store basic mass balance
        self.results['mass_balance'] = {
            'input_mass': input_mass,
            'dry_mass': dry_mass,
            'proteins': proteins,
            'lipids': lipids,
            'carbohydrates': carbohydrates
        }
        
        return self.results['mass_balance']

    def calculate_biogas_production(self) -> dict:
        """Estimate biogas production based on biochemical composition"""
        # Conversion factors (kg VS to m³ biogas)
        protein_yield = 0.49
        lipid_yield = 1.01
        carbohydrate_yield = 0.37

        vs_mass = self.results['mass_balance']['dry_mass']
        biogas = (
            self.waste_composition['P'] * protein_yield +
            self.waste_composition['L'] * lipid_yield +
            self.waste_composition['C'] * carbohydrate_yield
        ) * vs_mass

        self.results['biogas_production'] = {
            'total_biogas': biogas,
            'methane_content': 0.65,  # Typical methane fraction
            'digestate_output': vs_mass * 0.3  # 30% remains as digestate
        }
        return self.results['biogas_production']

    def calculate_energy_recovery(self) -> float:
        """Calculate energy potential from biogas"""
        methane_volume = self.results['biogas_production']['total_biogas'] * 0.65
        energy_mj = methane_volume * 36  # 36 MJ/m³ methane
        
        self.results['energy_recovery'] = {
            'total_energy': energy_mj,
            'recoverable_energy': energy_mj * 0.85  # Accounting for conversion losses
        }
        return self.results['energy_recovery']['recoverable_energy']

    def calculate_tea_metrics(self, tea_model) -> dict:
        """Calculate techno-economic analysis metrics

        Args:
            tea_model: Instance of AnaerobicDigestionTEA class

        Returns:
            Dictionary with economic metrics
        """
        input_mass = self.results['mass_balance']['input_mass']
        energy_output = self.results['energy_recovery']['recoverable_energy']
        
        capex = tea_model.capex_AD(input_mass)
        opex = tea_model.opex_AD(input_mass)
        energy_revenue = tea_model.calculate_energy_revenue(input_mass)[0]
        
        return {
            'capex': capex,
            'opex': opex,
            'energy_revenue': energy_revenue,
            'net_revenue': energy_revenue - (capex + opex)
        }

    def get_process_summary(self) -> dict:
        """Generate summary of process results for integration

        Returns:
            Dictionary formatted for integration with other components
        """
        return {
            'mass_balance': self.results['mass_balance'],
            'biogas_production': self.results['biogas_production'],
            'energy_recovery': self.results['energy_recovery'],
            'process_efficiency': self.results['energy_recovery']['recoverable_energy'] / 
            self.results['energy_recovery']['total_energy']
        }