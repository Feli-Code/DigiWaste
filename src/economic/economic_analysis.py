import streamlit as st
import numpy as np
import pandas as pd
from src.core.treatment.composting import CompostingModel

class AnaerobicDigestionTEA:
    def __init__(self, electricity_price=0.227585714, heat_price=0.075401667,
                 biomethane_injection_tariff=0.114, price_N=0.489868421,
                 price_P=0.734189189, price_K=0.553351351, compost_per_ton=50):
        """Initialize constants for the Anaerobic Digestion Plant Economic Analysis"""
        self.AD_CAPEX_COEFFICIENT = 35000
        self.AD_CAPEX_EXPONENT = 0.6
        self.AD_OPEX_COEFFICIENT = 17000
        self.AD_OPEX_EXPONENT = -0.6
        self.project_lifetime = 20

        self.CHP_COST_PER_MWH = 400000
        self.CHP_POWER_COEFFICIENT = 0.037421644

        self.BIOGAS_PRODUCTION_RATE = 137
        self.UPGRADER_COST_COEFFICIENT = 59788
        self.UPGRADER_COST_EXPONENT = -0.458
        self.MAX_UPGRADER_CAPACITY = 11200000
        self.ANNUAL_WORKING_HOURS = 8000

        self.DS_fraction = 0.045
        self.tot_N = 0.15
        self.tot_P = 0.007
        self.tot_K = 0.047
        self.Bio_N = 0.40
        self.fc_P = 0.22
        self.fc_K = 0.42

        self.price_N = price_N
        self.price_P = price_P
        self.price_K = price_K
        self.compost_per_ton = compost_per_ton

        self.digestate_conversion = 0.15 # from https://doi.org/10.3390/su152115644 even tho is written digestate is the quantity of final compost produced

        self.electricity_per_kg = 0.277
        self.heat_per_kg = 0.476
        self.biomethane_per_kg = 0.059184

        self.electricity_price = electricity_price
        self.heat_price = heat_price

        self.METHANE_FRACTION = 0.6
        self.HHV_methane = 10
        self.biomethane_injection_tariff = biomethane_injection_tariff

    def capex_AD(self, tons_treated):
        """Calculate total plant construction cost based on economy of scale."""
        capex_AD = self.AD_CAPEX_COEFFICIENT * (tons_treated ** self.AD_CAPEX_EXPONENT)
        return capex_AD

    def opex_AD(self, tons_treated):
        """Calculate total operating cost based on economy of scale."""
        opex_AD = self.AD_OPEX_COEFFICIENT * (tons_treated ** self.AD_OPEX_EXPONENT)
        return opex_AD

    def calculate_chp_cost(self, tons_treated):
        """Calculate Combined Heat and Power (CHP) unit cost."""
        chp_power_needed_kwh = self.CHP_POWER_COEFFICIENT * tons_treated
        chp_cost_total = (chp_power_needed_kwh / 1000) * self.CHP_COST_PER_MWH
        return chp_cost_total

    def calculate_biogas_upgrader_cost(self, tons_treated):
        """Calculate biogas upgrader cost."""
        total_gas_production = self.BIOGAS_PRODUCTION_RATE * tons_treated
        max_annual_capacity = self.MAX_UPGRADER_CAPACITY
        max_hourly_capacity = 1400

        num_upgraders = int(np.ceil(total_gas_production / max_annual_capacity))
        upgrader_cost_per_nmcubed_per_hour = self.UPGRADER_COST_COEFFICIENT * (max_hourly_capacity ** self.UPGRADER_COST_EXPONENT)
        total_upgrader_cost = num_upgraders * upgrader_cost_per_nmcubed_per_hour * max_hourly_capacity

        return total_upgrader_cost

    def calculate_total_cost(self, tons_treated):
        """Calculate the total cost of the anaerobic digestion plant."""
        capex_AD = self.capex_AD(tons_treated)
        opex_AD = self.opex_AD(tons_treated)
        total_chp_cost = self.calculate_chp_cost(tons_treated)
        total_plant_cost_CHP = capex_AD + (opex_AD * self.project_lifetime)

        total_upgrader_cost = self.calculate_biogas_upgrader_cost(tons_treated)
        total_plant_cost_Upgrade = total_plant_cost_CHP - total_chp_cost + total_upgrader_cost

        cost_per_ton_CHP = total_plant_cost_CHP / (tons_treated * self.project_lifetime)
        cost_per_ton_Upgrade = total_plant_cost_Upgrade / (tons_treated * self.project_lifetime)

        return total_plant_cost_CHP, total_plant_cost_Upgrade, cost_per_ton_CHP, cost_per_ton_Upgrade

    def calculate_digestate_value(self, tons_treated):
        """Calculate the monetary value (in €) of substituted mineral fertilizers."""
        if self.compost_per_ton > 0:
            return self.compost_per_ton * (tons_treated*self.digestate_conversion)
        else:
            D_m = self.digestate_conversion * tons_treated  # convert tons to kg
            N_fert = D_m * self.DS_fraction * self.tot_N * self.Bio_N
            P_fert = D_m * self.DS_fraction * self.tot_P * self.fc_P
            K_fert = D_m * self.DS_fraction * self.tot_K * self.fc_K

            N_value = N_fert * self.price_N
            P_value = P_fert * self.price_P
            K_value = K_fert * self.price_K

            total_value = N_value + P_value + K_value
            return total_value

    def calculate_energy_revenue(self, tons_treated):
        """Calculate the revenue from selling electricity and heat."""
        kg_biowaste = tons_treated * 1000
        electricity_kwh = kg_biowaste * self.electricity_per_kg
        heat_kwh = kg_biowaste * self.heat_per_kg

        electricity_revenue = electricity_kwh * self.electricity_price
        heat_revenue = heat_kwh * self.heat_price
        total_energy_revenue = electricity_revenue + heat_revenue
        return total_energy_revenue, electricity_revenue, heat_revenue

    def calculate_biomethane_injection_revenue(self, tons_treated):
        """Calculate the revenue from biomethane injection into the grid."""
        m3_biogas = self.BIOGAS_PRODUCTION_RATE * tons_treated
        m3_methane = m3_biogas * self.METHANE_FRACTION
        energy_methane = m3_methane * self.HHV_methane  # in kWh HHV
        revenue = energy_methane * self.biomethane_injection_tariff
        return revenue

class IncinerationTEA:
    def __init__(self, electricity_price=0.227585714, heat_price=0.075401667):
        """Initialize constants for the Incineration Economic Analysis"""
        self.INC_CAPEX_COEFFICIENT = 5000
        self.INC_CAPEX_EXPONENT = 0.8
        self.INC_OPEX_COEFFICIENT = 700
        self.INC_OPEX_EXPONENT = -0.3
        self.project_lifetime = 20
        self.INCINERATION_TAX = 7.5  # €/ton

        self.electricity_per_kg_INC = 0.04513
        self.heat_per_kg_INC = 0.765000612

        self.electricity_price = electricity_price
        self.heat_price = heat_price

    def capex_INC(self, tons_treated):
        capex_INC = self.INC_CAPEX_COEFFICIENT * (tons_treated ** self.INC_CAPEX_EXPONENT)
        return capex_INC

    def opex_INC(self, tons_treated):
        opex_INC = self.INC_OPEX_COEFFICIENT * (tons_treated ** self.INC_OPEX_EXPONENT)
        incineration_tax = self.INCINERATION_TAX * tons_treated
        return opex_INC + incineration_tax

    def calculate_total_cost(self, tons_treated):
        capex_INC = self.capex_INC(tons_treated)
        opex_INC = self.opex_INC(tons_treated)
        total_plant_cost_INC = capex_INC + (opex_INC * self.project_lifetime)
        cost_per_ton_INC = total_plant_cost_INC / (tons_treated * self.project_lifetime)
        return total_plant_cost_INC, cost_per_ton_INC

    def calculate_energy_revenue(self, tons_treated):
        kg_biowaste = tons_treated * 1000
        electricity_kwh_INC = kg_biowaste * self.electricity_per_kg_INC
        heat_kwh_INC = kg_biowaste * self.heat_per_kg_INC

        electricity_revenue_INC = electricity_kwh_INC * self.electricity_price
        heat_revenue_INC = heat_kwh_INC * self.heat_price
        total_energy_revenue_INC = electricity_revenue_INC + heat_revenue_INC
        return total_energy_revenue_INC, electricity_revenue_INC, heat_revenue_INC

class CompostingTEA:
    def __init__(self, price_N=489868421, price_P=734189189, price_K=553351351, compost_per_ton=50):
        """Initialize constants for the Composting Economic Analysis
        
        Args:
            price_N: Price of nitrogen fertilizer (€/ton N)
            price_P: Price of phosphorus fertilizer (€/ton P)
            price_K: Price of potassium fertilizer (€/ton K)
            compost_per_ton: Fixed price per ton of compost if using market value
        """
        # Cost coefficients from Eunomia data
        self.COMPOSTING_COST_COEFFICIENT = 55.412
        self.COMPOSTING_COST_COEFFICIENT_2 = -28151
        
        # Initialize with default values but allow override from composting model
        self.degradation_coefficient = 0.33  # Will be updated from actual organic matter degradation
        self.moisture_content = None  # Will be set from composting model output
        self.dry_matter_content = None  # Will be calculated from moisture content
        
        # Nutrient contents (to be updated from waste characteristics)
        self.tot_N = 0.012  # Total nitrogen content
        self.tot_P = 0.0093  # Total phosphorus content
        self.tot_K = 0.011  # Total potassium content
        
        # Bioavailability factors
        self.Bio_N = 0.30  # Nitrogen bioavailability
        self.fc_P = 0.22   # Phosphorus conversion factor
        self.fc_K = 0.42   # Potassium conversion factor
        
        # Market prices
        self.price_N = price_N
        self.price_P = price_P
        self.price_K = price_K
        self.compost_per_ton = compost_per_ton
        
        # Process parameters (to be updated from model)
        self.energy_consumption = 0
        self.water_consumption = 0
        self.emissions = {}
        
    def update_from_composting_model(self, composting_model):
        """Update TEA parameters from composting model results
        
        Args:
            composting_model: Instance of CompostingModel with simulation results
        """
        if hasattr(composting_model, 'df_combined'):
            # Update degradation from actual organic matter loss
            final_om = composting_model.df_combined[['C', 'CE', 'H', 'LG', 'L']].iloc[-1].sum()
            initial_om = composting_model.df_combined[['C', 'CE', 'H', 'LG', 'L']].iloc[0].sum()
            self.degradation_coefficient = 1 - (final_om / initial_om)
            
            # Update moisture and dry matter content
            self.moisture_content = composting_model.df_combined['Mc'].iloc[-1] / composting_model.TM_0
            self.dry_matter_content = 1 - self.moisture_content
            
            # Update nutrient contents from waste characteristics if available
            if hasattr(composting_model, 'waste_characteristics'):
                self.tot_N = composting_model.waste_characteristics.get('nitrogen_content', self.tot_N)
                self.tot_P = composting_model.waste_characteristics.get('phosphorus_content', self.tot_P)
                self.tot_K = composting_model.waste_characteristics.get('potassium_content', self.tot_K)
            
            # Update process parameters
            self.energy_consumption = composting_model.df_combined.get('energy_consumption', 0)
            self.water_consumption = composting_model.df_combined.get('water_consumption', 0)
            self.emissions = composting_model.cumulative_emissions if hasattr(composting_model, 'cumulative_emissions') else {}


    def calculate_composting_cost(self, tons_treated):
        """Calculate total composting cost including operational adjustments
        
        Args:
            tons_treated: Amount of waste input in metric tons
            
        Returns:
            Total cost including base cost and operational adjustments
        """
        # Base cost from scale economy
        base_cost = (self.COMPOSTING_COST_COEFFICIENT * tons_treated) + self.COMPOSTING_COST_COEFFICIENT_2
        
        # Adjust cost based on process parameters if available
        cost_adjustments = 0
        if self.energy_consumption:
            cost_adjustments += self.energy_consumption * 0.15  # Energy cost adjustment
        if self.water_consumption:
            cost_adjustments += self.water_consumption * 0.02  # Water cost adjustment
            
        return base_cost + cost_adjustments

    def calculate_total_cost(self, tons_treated):
        """Calculate total cost and cost per ton including operational costs
        
        Args:
            tons_treated: Amount of waste input in metric tons
            
        Returns:
            Tuple of (total cost, cost per ton)
        """
        total_composting_cost = self.calculate_composting_cost(tons_treated)
        
        # Calculate cost per ton over project lifetime
        cost_per_ton = total_composting_cost / (tons_treated * 20)
        return total_composting_cost, cost_per_ton

    def calculate_compost_value(self, tons_treated):
        """Calculate the value of produced compost based on nutrient content or market price
        
        Args:
            tons_treated: Amount of waste input in metric tons
            
        Returns:
            Total value of produced compost
        """
        if self.compost_per_ton > 0:
            # Use market price if specified
            compost_mass = tons_treated * self.degradation_coefficient
            return self.compost_per_ton * compost_mass
        else:
            # Calculate value based on nutrient content
            compost_mass = tons_treated * self.degradation_coefficient
            dry_mass = compost_mass * (1 - self.moisture_content if self.moisture_content is not None else self.dry_matter_content)
            
            # Calculate nutrient quantities
            N_fert = dry_mass * self.tot_N * self.Bio_N
            P_fert = dry_mass * self.tot_P * self.fc_P
            K_fert = dry_mass * self.tot_K * self.fc_K
            
            # Calculate nutrient values
            N_value = N_fert * self.price_N
            P_value = P_fert * self.price_P
            K_value = K_fert * self.price_K
            
            return N_value + P_value + K_value

class NPVAnalysis:
    def __init__(self, discount_rate=0.05):
        self.discount_rate = discount_rate

    def calculate_npv(self, initial_investment, annual_cash_flows, project_lifetime=20):
        if len(annual_cash_flows) < project_lifetime:
            annual_cash_flows = list(annual_cash_flows) + [annual_cash_flows[-1]] * (project_lifetime - len(annual_cash_flows))

        discounted_cash_flows = [
            cash_flow / ((1 + self.discount_rate) ** year)
            for year, cash_flow in enumerate([-initial_investment] + annual_cash_flows, start=0)
        ]

        return sum(discounted_cash_flows)

def run_economic_analysis():
    st.title("Waste Treatment Economic Analysis App")

    with st.sidebar:
        st.header("Input Parameters")
        tons_input = st.text_input("Tons treated (comma-separated)", "2500,5000,10000,50000,100000,440000")
        tax_rate = st.number_input("Collection Tax (€/t)", value=19.0)
        discount_rate = st.number_input("Discount Rate", value=0.05, format="%.2f")

        st.subheader("Electricity and Heat Prices")
        electricity_price = st.number_input("Electricity Price (€/kWh)", value=0.227585714)
        heat_price = st.number_input("Heat Price (€/kWh)", value=0.075401667)

        st.subheader("Biomethane Tariff")
        biomethane_tariff = st.number_input("Biomethane Tariff (€/kWh)", value=0.114)

        st.subheader("Fertilizer Prices (€/)")
        price_N = st.number_input("Nitrogen Price", value=489.868421)
        price_P = st.number_input("Phosphorus Price", value=734.189189)
        price_K = st.number_input("Potassium Price", value=553.351351)

        st.subheader("Composting and Digestate Cost")
        use_fixed_cost = st.checkbox("Use fixed cost per ton for composting and digestate")
        cost_per_ton = 50 if use_fixed_cost else 0

        run_analysis = st.button("Run Analysis")

    if run_analysis:
        try:
            tons_range = [int(x.strip()) for x in tons_input.split(',')]
        except:
            st.error("Invalid input for tons treated. Please enter comma-separated numbers.")
            return

        TEA_AD = AnaerobicDigestionTEA(electricity_price, heat_price, biomethane_tariff, price_N, price_P, price_K, cost_per_ton)
        TEA_incineration = IncinerationTEA(electricity_price, heat_price)
        TEA_composting = CompostingTEA(price_N, price_P, price_K, cost_per_ton)
        npv_analyzer = NPVAnalysis(discount_rate)

        cost_data = []
        npv_data = []
        for tons in tons_range:
            cost_plant_CHP, cost_plant_Upgrade, cost_per_ton_CHP, cost_per_ton_Upgrade = TEA_AD.calculate_total_cost(tons)
            cost_inc, cost_per_ton_inc = TEA_incineration.calculate_total_cost(tons)
            cost_comp, cost_per_ton_comp = TEA_composting.calculate_total_cost(tons)

            cost_data.append({
                "Tons Treated (t)": tons,
                "AD CHP Cost (€/t)": cost_per_ton_CHP,
                "AD Upgrade Cost (€/t)": cost_per_ton_Upgrade,
                "Incineration Cost (€/t)": cost_per_ton_inc,
                "Composting Cost (€/t)": cost_per_ton_comp
            })

            compost_value = TEA_composting.calculate_compost_value(tons)
            collection_tax = tax_rate * tons  # additional revenue per year from collection tax

            digestate_value = TEA_AD.calculate_digestate_value(tons)
            total_energy_revenue_AD = TEA_AD.calculate_energy_revenue(tons)[0]
            biomethane_injection_revenue = TEA_AD.calculate_biomethane_injection_revenue(tons)

            # --- AD with CHP ---
            annual_cash_flow_AD_CHP = total_energy_revenue_AD + digestate_value + collection_tax - (cost_plant_CHP / 20)
            annual_cash_flows_AD_CHP = [annual_cash_flow_AD_CHP] * 20
            npv_AD_CHP = npv_analyzer.calculate_npv(
                initial_investment=cost_plant_CHP,
                annual_cash_flows=annual_cash_flows_AD_CHP
            )

            # --- AD with Upgrading ---
            annual_cash_flow_AD_Upgrade = digestate_value + biomethane_injection_revenue + collection_tax - (cost_plant_Upgrade / 20)
            annual_cash_flows_AD_Upgrade = [annual_cash_flow_AD_Upgrade] * 20
            npv_AD_Upgrade = npv_analyzer.calculate_npv(
                initial_investment=cost_plant_Upgrade,
                annual_cash_flows=annual_cash_flows_AD_Upgrade
            )

            total_energy_revenue_INC = TEA_incineration.calculate_energy_revenue(tons)[0]
            annual_cash_flow_INC = total_energy_revenue_INC + collection_tax - (cost_inc / 20)
            annual_cash_flows_INC = [annual_cash_flow_INC] * 20
            npv_INC = npv_analyzer.calculate_npv(
                initial_investment=cost_inc,
                annual_cash_flows=annual_cash_flows_INC
            )

            annual_cash_flow_composting = compost_value + collection_tax - (cost_comp / 20)
            annual_cash_flows_composting = [annual_cash_flow_composting] * 20
            npv_composting = npv_analyzer.calculate_npv(
                initial_investment=cost_comp,
                annual_cash_flows=annual_cash_flows_composting
            )

            npv_data.append({
                "Tons Treated (t)": tons,
                "NPV AD with CHP (€)": npv_AD_CHP,
                "NPV AD with Upgrading (€)": npv_AD_Upgrade,
                "NPV Incineration (€)": npv_INC,
                "NPV Composting (€)": npv_composting,
            })

        cost_df = pd.DataFrame(cost_data)
        npv_df = pd.DataFrame(npv_data)

        st.header("Cost Analysis per Ton")
        st.dataframe(cost_df.style.format("{:.2f}"), use_container_width=True)

        st.header("Cost Comparison")
        cost_chart_df = cost_df.melt(id_vars="Tons Treated (t)",
                                   value_vars=["AD CHP Cost (€/t)", "AD Upgrade Cost (€/t)",
                                              "Incineration Cost (€/t)", "Composting Cost (€/t)"],
                                   var_name="Technology", value_name="Cost")
        st.line_chart(cost_chart_df, x="Tons Treated (t)", y="Cost", color="Technology", use_container_width=True)

        st.header("NPV Analysis")
        st.dataframe(npv_df.style.format("{:.2f}"), use_container_width=True)

        st.header("NPV Comparison")
        npv_chart_df = npv_df.melt(id_vars="Tons Treated (t)",
                                   value_vars=["NPV AD with CHP (€)", "NPV AD with Upgrading (€)",
                                              "NPV Incineration (€)", "NPV Composting (€)"],
                                   var_name="Technology", value_name="NPV")
        st.line_chart(npv_chart_df, x="Tons Treated (t)", y="NPV", color="Technology", use_container_width=True)
