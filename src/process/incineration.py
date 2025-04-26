class IncinerationModel:
    """Incineration process model for biowaste treatment

    Handles mass balance calculations, energy recovery estimation,
    and emission tracking for waste incineration processes.
    """

    def __init__(self, waste_composition: dict):
        """Initialize incineration model parameters

        Args:
            waste_composition: Dictionary containing waste characteristics
                Expected keys: moisture_content, carbon_content, inert_content
        """
        self.waste_composition = waste_composition
        self.results = {
            'mass_balance': {},
            'energy_recovery': {},
            'emissions': {}
        }

    def calculate_mass_balance(self, input_mass: float) -> dict:
        """Calculate mass balance for incineration process

        Args:
            input_mass: Total mass of waste input (kg)

        Returns:
            Dictionary with mass balance components
        """
        # Calculate moisture evaporation
        moisture_loss = input_mass * self.waste_composition['moisture_content']
        
        # Calculate combustible mass (organic + fixed carbon)
        combustible_mass = input_mass * (1 - self.waste_composition['inert_content'])
        
        # Calculate ash residues
        ash_residue = input_mass * self.waste_composition['inert_content']
        
        # Store results
        self.results['mass_balance'] = {
            'input_mass': input_mass,
            'moisture_loss': moisture_loss,
            'combustible_mass': combustible_mass,
            'ash_residue': ash_residue
        }
        
        return self.results['mass_balance']

    def calculate_energy_recovery(self, calorific_value: float) -> float:
        """Estimate energy recovery from combustion

        Args:
            calorific_value: Calorific value of waste (MJ/kg)

        Returns:
            Total recoverable energy (MJ)
        """
        combustible_mass = self.results['mass_balance']['combustible_mass']
        energy = combustible_mass * calorific_value
        
        # Assume 25% conversion efficiency for heat recovery
        recoverable_energy = energy * 0.25
        
        self.results['energy_recovery'] = {
            'total_energy': energy,
            'recoverable_energy': recoverable_energy,
            'efficiency': 0.25
        }
        
        return recoverable_energy

    def calculate_emissions(self) -> dict:
        """Estimate key emissions from incineration process

        Returns:
            Dictionary with emission values (kg)
        """
        combustible_mass = self.results['mass_balance']['combustible_mass']
        
        # Emission factors (kg/kg waste)
        co2_factor = 1.5 * self.waste_composition['carbon_content']
        
        self.results['emissions'] = {
            'CO2': combustible_mass * co2_factor,
            'Ash': self.results['mass_balance']['ash_residue']
        }
        
        return self.results['emissions']

    def calculate_tea_metrics(self, tea_model) -> dict:
        """Calculate techno-economic analysis metrics

        Args:
            tea_model: Instance of IncinerationTEA class

        Returns:
            Dictionary with economic metrics
        """
        input_mass = self.results['mass_balance']['input_mass']
        energy_output = self.results['energy_recovery']['recoverable_energy']
        
        # Calculate costs and revenues
        capex = tea_model.capex_INC(input_mass)
        opex = tea_model.opex_INC(input_mass)
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
            'energy_recovery': self.results['energy_recovery'],
            'emissions': self.results['emissions'],
            'process_efficiency': self.results['energy_recovery']['efficiency']
        }
