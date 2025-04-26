import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.normalization import Normalization
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.core.economic.economic_analysis import (
    AnaerobicDigestionTEA,
    IncinerationTEA,
    CompostingTEA,
    NPVAnalysis
)

# Environmental impacts (sum of HHI, EI, RI) for each technology
ENV_IMPACT = {
    'Incineration': 0.242e-1 + (-6.41e-2) + (-4.30e-2),
    'AD_CHP': -(-2.53e-1 + (-1.63e-1) + (-4.10e-2)),
    'AD_Upgrade': -(-2.53e-1 + (-1.63e-1) + (-4.10e-2)),
    'Composting': 5.13e-2 + 2.67e-2 + 5.28e-3
}

# Scaling factor for NPV (to express in millions of euros)
NPV_SCALE = 1e6

class WasteTreatmentProblem(ElementwiseProblem):
    def __init__(self, total_waste, params, constraints=None):
        self.total_waste = total_waste
        self.params = params
        self.constraints = constraints or {}

        # Define parameters for each TEA class
        self.inc_params = {
            'electricity_price': params.get('electricity_price'),
            'heat_price': params.get('heat_price')
        }

        self.ad_params = {
            'electricity_price': params.get('electricity_price'),
            'heat_price': params.get('heat_price'),
            'biomethane_injection_tariff': params.get('biomethane_tariff'),
            'price_N': params.get('price_N'),
            'price_P': params.get('price_P'),
            'price_K': params.get('price_K'),
            'cost_per_ton': params.get('cost_per_ton')
        }

        self.comp_params = {
            'price_N': params.get('price_N'),
            'price_P': params.get('price_P'),
            'price_K': params.get('price_K'),
            'compost_per_ton': params.get('compost_per_ton')
        }

        # Remove None values if any
        self.inc_params = {k: v for k, v in self.inc_params.items() if v is not None}
        self.ad_params = {k: v for k, v in self.ad_params.items() if v is not None}
        self.comp_params = {k: v for k, v in self.comp_params.items() if v is not None}

        # Count how many additional constraints we have (min/max for each technology)
        n_additional_constraints = sum(1 for k in self.constraints if k.startswith(('min_', 'max_')))

        super().__init__(
            n_var=4,  # Decision variables [Incineration, AD_CHP, AD_Upgrade, Composting]
            n_obj=2 if params.get('npv_weight') is None else 1,  # Objectives: maximize NPV (minimize -NPV) and minimize Environmental Impact
            n_constr=1 + n_additional_constraints,  # Total waste + additional constraints
            xl=0,  # Lower bound
            xu=total_waste  # Upper bound
        )

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            x_inc, x_ad_chp, x_ad_up, x_comp = x

            # Primary constraint: sum of allocations must equal total waste
            constraints = [sum(x) - self.total_waste]

            # Add technology-specific constraints
            if 'min_incineration' in self.constraints:
                constraints.append(self.constraints['min_incineration'] - x_inc)
            if 'max_incineration' in self.constraints:
                constraints.append(x_inc - self.constraints['max_incineration'])

            if 'min_ad_chp' in self.constraints:
                constraints.append(self.constraints['min_ad_chp'] - x_ad_chp)
            if 'max_ad_chp' in self.constraints:
                constraints.append(x_ad_chp - self.constraints['max_ad_chp'])

            if 'min_ad_upgrade' in self.constraints:
                constraints.append(self.constraints['min_ad_upgrade'] - x_ad_up)
            if 'max_ad_upgrade' in self.constraints:
                constraints.append(x_ad_up - self.constraints['max_ad_upgrade'])

            if 'min_composting' in self.constraints:
                constraints.append(self.constraints['min_composting'] - x_comp)
            if 'max_composting' in self.constraints:
                constraints.append(x_comp - self.constraints['max_composting'])

            out["G"] = constraints

            # Calculate NPV for each technology
            npv_inc = self._calculate_incineration_npv(x_inc)
            npv_ad_chp = self._calculate_ad_chp_npv(x_ad_chp)
            npv_ad_up = self._calculate_ad_upgrade_npv(x_ad_up)
            npv_comp = self._calculate_compost_npv(x_comp)

            total_npv = npv_inc + npv_ad_chp + npv_ad_up + npv_comp
            scaled_total_npv = total_npv / NPV_SCALE

            # Calculate environmental impact (weighted by tons allocated)
            env_impact = (
                ENV_IMPACT['Incineration'] * x_inc +
                ENV_IMPACT['AD_CHP'] * x_ad_chp +
                ENV_IMPACT['AD_Upgrade'] * x_ad_up +
                ENV_IMPACT['Composting'] * x_comp
            )

            # Apply objective weights if provided
            if 'npv_weight' in self.params and 'env_weight' in self.params:
                npv_weight = self.params['npv_weight']
                env_weight = self.params['env_weight']

                # Normalize weights
                sum_weights = npv_weight + env_weight
                npv_weight = npv_weight / sum_weights
                env_weight = env_weight / sum_weights

                # Normalize objectives
                norm_npv = scaled_total_npv / 100.0  # Assuming NPVs are typically in the range of 0-100 million
                norm_env = env_impact / 50000        # Assuming env impacts are typically in the range of -10k to 50k

                # Calculate weighted objective
                weighted_obj = -npv_weight * norm_npv + env_weight * norm_env
                out["F"] = [weighted_obj]
            else:
                out["F"] = [-scaled_total_npv, env_impact]  # Default to maximizing NPV and minimizing environmental impact

        except Exception as e:
            print(f"Error in _evaluate: {e} for x={x}")
            out["F"] = [1e10] if 'npv_weight' in self.params else [1e10, 1e10]
            out["G"] = [1e10] * len(self.constraints)

    def _calculate_incineration_npv(self, tons):
        if tons <= 0:
            return 0
        tea = IncinerationTEA(**self.inc_params)
        capex = tea.capex_INC(tons)
        opex = tea.opex_INC(tons)
        revenue = tea.calculate_energy_revenue(tons)[0]
        collection_tax = self.params['tax_rate'] * tons

        annual_cf = revenue + collection_tax - (capex + opex * 20) / 20
        total_investment = capex + opex * 20

        return NPVAnalysis(self.params['discount_rate']).calculate_npv(
            total_investment,
            [annual_cf] * 20
        )

    def _calculate_ad_chp_npv(self, tons):
        if tons <= 0:
            return 0
        tea = AnaerobicDigestionTEA(**self.ad_params)
        capex = tea.capex_AD(tons)
        opex = tea.opex_AD(tons)
        chp_cost = tea.calculate_chp_cost(tons)
        total_cost = capex + opex * 20

        energy_rev = tea.calculate_energy_revenue(tons)[0]
        digestate = tea.calculate_digestate_value(tons)
        tax = self.params['tax_rate'] * tons

        annual_cf = energy_rev + digestate + tax - total_cost / 20

        return NPVAnalysis(self.params['discount_rate']).calculate_npv(
            total_cost,
            [annual_cf] * 20
        )

    def _calculate_ad_upgrade_npv(self, tons):
        if tons <= 0:
            return 0
        tea = AnaerobicDigestionTEA(**self.ad_params)
        capex = tea.capex_AD(tons)
        opex = tea.opex_AD(tons)
        chp_cost = tea.calculate_chp_cost(tons)
        upgrader_cost = tea.calculate_biogas_upgrader_cost(tons)
        total_cost = capex + opex * 20 - chp_cost + upgrader_cost

        digestate = tea.calculate_digestate_value(tons)
        biomethane = tea.calculate_biomethane_injection_revenue(tons)
        tax = self.params['tax_rate'] * tons

        annual_cf = digestate + biomethane + tax - total_cost / 20

        return NPVAnalysis(self.params['discount_rate']).calculate_npv(
            total_cost,
            [annual_cf] * 20
        )

    def _calculate_compost_npv(self, tons):
        if tons <= 0:
            return 0
        tea = CompostingTEA(**self.comp_params)
        total_cost, _ = tea.calculate_total_cost(tons)
        compost_value = tea.calculate_compost_value(tons)
        tax = self.params['tax_rate'] * tons

        annual_cf = compost_value + tax - total_cost / 20

        return NPVAnalysis(self.params['discount_rate']).calculate_npv(
            total_cost,
            [annual_cf] * 20
        )

def optimize_waste_allocation(total_waste, tea_params, optimization_params, decision_constraints=None, mode="pareto"):
    params = {**tea_params, **optimization_params}
    problem = WasteTreatmentProblem(total_waste, params, constraints=decision_constraints)

    if mode == "weighted":
        algorithm = NSGA2(pop_size=100)
    else:
        algorithm = NSGA2(pop_size=100, normalize=Normalization())

    try:
        res = minimize(problem, algorithm, ('n_gen', 100), verbose=True)
        if hasattr(res, 'X') and hasattr(res, 'F') and res.X is not None and res.F is not None:
            # Ensure solutions is a 2D array
            solutions = res.X
            if solutions.ndim == 1:
                solutions = solutions.reshape(1, -1)  # Reshape to 2D if single solution
            objectives = res.F

            if mode == "weighted":
                npv_values = []
                env_values = []

                for x in solutions:
                    x_inc, x_ad_chp, x_ad_up, x_comp = x

                    # Calculate NPV for each solution
                    npv_inc = problem._calculate_incineration_npv(x_inc)
                    npv_ad_chp = problem._calculate_ad_chp_npv(x_ad_chp)
                    npv_ad_up = problem._calculate_ad_upgrade_npv(x_ad_up)
                    npv_comp = problem._calculate_compost_npv(x_comp)

                    total_npv = npv_inc + npv_ad_chp + npv_ad_up + npv_comp
                    scaled_total_npv = total_npv / NPV_SCALE

                    # Calculate environmental impact
                    env_impact = (
                        ENV_IMPACT['Incineration'] * x_inc +
                        ENV_IMPACT['AD_CHP'] * x_ad_chp +
                        ENV_IMPACT['AD_Upgrade'] * x_ad_up +
                        ENV_IMPACT['Composting'] * x_comp
                    )

                    npv_values.append(scaled_total_npv)
                    env_values.append(env_impact)

                objectives = np.column_stack([npv_values, env_values])
            else:
                # Convert back to positive NPV (in millions)
                objectives[:, 0] = -objectives[:, 0]

            return solutions, objectives
        else:
            print("Warning: Optimization did not produce valid results")
            return None, None
    except Exception as e:
        print(f"Error during optimization: {e}")
        return None, None

def visualize_solution(solution, total_waste, solution_id=None):
    """Visualize a single solution as a pie chart"""
    labels = ["Incineration", "AD with CHP", "AD with Upgrading", "Composting"]
    values = solution

    solution_df = pd.DataFrame({
        "Technology": labels,
        "Allocation (tons)": values,
        "Percentage (%)": (values / total_waste * 100).round(1)
    })

    # Create a title based on the solution ID
    title = f"Solution {solution_id}" if solution_id else "Selected Solution"

    # Display the numerical values
    st.write(f"### {title} - Allocation Details")
    st.dataframe(solution_df)

    # Create the pie chart
    pie_data = pd.DataFrame({
        'Technology': labels,
        'Allocation': values
    })

    pie_chart = alt.Chart(pie_data).mark_arc().encode(
        theta=alt.Theta(field="Allocation", type="quantitative"),
        color=alt.Color(field="Technology", type="nominal", scale=alt.Scale(scheme='category10')),
        tooltip=['Technology', 'Allocation']
    ).properties(
        title=f"{title} - Visualization",
        width=350,
        height=350
    )

    st.altair_chart(pie_chart, use_container_width=True)

def run_optimization_analysis():
    """Main function to run the waste treatment optimization analysis"""
    # Create a multi-tab interface
    tab1, tab2, tab3 = st.tabs(["Parameter Settings", "Decision Maker Preferences", "Results & Analysis"])

    with tab1:
        st.header("Waste and Economic Parameters")

        col1, col2 = st.columns(2)
        with col1:
            total_waste = st.number_input("Total Waste (tons)", value=100000, step=1000)
            tax_rate = st.number_input("Collection Tax (€/t)", value=19.0)
            discount_rate = st.number_input("Discount Rate", value=0.05, format="%.2f")
            electricity_price = st.number_input("Electricity Price (€/kWh)", value=0.227)
            heat_price = st.number_input("Heat Price (€/kWh)", value=0.075)

        with col2:
            biomethane_tariff = st.number_input("Biomethane Tariff (€/kWh)", value=0.114)
            price_N = st.number_input("Nitrogen Price (€/t)", value=489)
            price_P = st.number_input("Phosphorus Price (€/t)", value=734)
            price_K = st.number_input("Potassium Price (€/t)", value=553)
            use_fixed_cost = st.checkbox("Use fixed cost per ton", value=True)
            compost_per_ton = st.number_input("Compost Cost (€/t)", value=50) if use_fixed_cost else 0

    # Define session state to store optimization results
    if 'solutions' not in st.session_state:
        st.session_state.solutions = None
        st.session_state.objectives = None
        st.session_state.selected_solution = None
        st.session_state.optimization_mode = "pareto"

    with tab2:
        st.header("Decision Maker Preferences")

        # Decision making approach
        st.subheader("Optimization Approach")
        optimization_method = st.radio(
            "Select optimization method:",
            ["Multi-objective (Pareto frontier)", "Weighted-sum (single solution)"],
            index=0
        )

        if optimization_method == "Weighted-sum (single solution)":
            st.session_state.optimization_mode = "weighted"
            col1, col2 = st.columns(2)
            with col1:
                npv_weight = st.slider("Economic (NPV) Weight", 0.1, 10.0, 1.0, 0.1)
            with col2:
                env_weight = st.slider("Environmental Impact Weight", 0.1, 10.0, 1.0, 0.1)
        else:
            st.session_state.optimization_mode = "pareto"
            npv_weight = None
            env_weight = None

        # Technology constraints
        st.subheader("Technology Constraints")
        st.write("Specify minimum and maximum allocation for each technology (tons):")

        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Minimum Allocations")
            min_incineration = st.number_input("Min. Incineration (tons)", 0, int(total_waste), 0)
            min_ad_chp = st.number_input("Min. AD with CHP (tons)", 0, int(total_waste), 0)
            min_ad_upgrade = st.number_input("Min. AD with Upgrading (tons)", 0, int(total_waste), 0)
            min_composting = st.number_input("Min. Composting (tons)", 0, int(total_waste), 0)

        with col2:
            st.write("#### Maximum Allocations")
            max_incineration = st.number_input("Max. Incineration (tons)", 0, int(total_waste), int(total_waste))
            max_ad_chp = st.number_input("Max. AD with CHP (tons)", 0, int(total_waste), int(total_waste))
            max_ad_upgrade = st.number_input("Max. AD with Upgrading (tons)", 0, int(total_waste), int(total_waste))
            max_composting = st.number_input("Max. Composting (tons)", 0, int(total_waste), int(total_waste))

        # Check if the constraints are feasible
        total_min = min_incineration + min_ad_chp + min_ad_upgrade + min_composting
        total_max = max_incineration + max_ad_chp + max_ad_upgrade + max_composting

        if total_min > total_waste:
            st.error(f"The sum of minimum allocations ({total_min} tons) exceeds the total waste ({total_waste} tons). Please adjust your constraints.")

        if total_max < total_waste:
            st.error(f"The sum of maximum allocations ({total_max} tons) is less than the total waste ({total_waste} tons). Please adjust your constraints.")

        # Collect constraints
        constraints = {}
        if min_incineration > 0:
            constraints['min_incineration'] = min_incineration
        if max_incineration < total_waste:
            constraints['max_incineration'] = max_incineration

        if min_ad_chp > 0:
            constraints['min_ad_chp'] = min_ad_chp
        if max_ad_chp < total_waste:
            constraints['max_ad_chp'] = max_ad_chp

        if min_ad_upgrade > 0:
            constraints['min_ad_upgrade'] = min_ad_upgrade
        if max_ad_upgrade < total_waste:
            constraints['max_ad_upgrade'] = max_ad_upgrade

        if min_composting > 0:
            constraints['min_composting'] = min_composting
        if max_composting < total_waste:
            constraints['max_composting'] = max_composting

        # Run optimization button
        if st.button("Run Optimization", key="run_optimization_btn"):
            tea_params = {
                'electricity_price': electricity_price,
                'heat_price': heat_price,
                'biomethane_tariff': biomethane_tariff,
                'price_N': price_N,
                'price_P': price_P,
                'price_K': price_K,
                'compost_per_ton': compost_per_ton
            }

            optimization_params = {
                'discount_rate': discount_rate,
                'tax_rate': tax_rate
            }

            # Add weights if using weighted-sum approach
            if st.session_state.optimization_mode == "weighted":
                optimization_params['npv_weight'] = npv_weight
                optimization_params['env_weight'] = env_weight

            with st.spinner("Running optimization..."):
                solutions, objectives = optimize_waste_allocation(
                    total_waste,
                    tea_params,
                    optimization_params,
                    decision_constraints=constraints,
                    mode=st.session_state.optimization_mode
                )

                if solutions is None or objectives is None:
                    st.error("Optimization failed. Please check parameters and try again.")
                else:
                    st.session_state.solutions = solutions
                    st.session_state.objectives = objectives
                    st.success("Optimization completed successfully!")

    with tab3:
        st.header("Results & Analysis")

        if st.session_state.solutions is not None and st.session_state.objectives is not None:
            solutions = st.session_state.solutions
            objectives = st.session_state.objectives

            # Assigning an ID to each solution
            solution_ids = np.arange(len(solutions)) + 1

            # Create DataFrame for solutions
            allocation_data = pd.DataFrame(solutions, columns=["Incineration", "AD with CHP", "AD with Upgrading", "Composting"])
            allocation_data["Solution ID"] = solution_ids

            # Create DataFrame for Pareto front
            objective_data = pd.DataFrame({
                "Solution ID": solution_ids,
                "NPV (Million €)": objectives[:, 0],
                "Environmental Impact": objectives[:, 1]
            })

            # Sort solutions by NPV
            sorted_indices = np.argsort(-objective_data["NPV (Million €)"].values)
            sorted_objective_data = objective_data.iloc[sorted_indices].copy()
            sorted_allocation_data = allocation_data.iloc[sorted_indices].copy()

            # Interactive Pareto Front Chart
            st.subheader("Solution Space Visualization")

            # Calculate mid-point solution as a default
            if len(sorted_objective_data) > 0:
                default_solution_index = len(sorted_objective_data) // 2
                default_solution_id = sorted_objective_data.iloc[default_solution_index]["Solution ID"]
            else:
                default_solution_id = 1

            # Interactive chart
            scatter = alt.Chart(sorted_objective_data).mark_circle(size=80).encode(
                x=alt.X("NPV (Million €):Q", title="NPV (Million €)", scale=alt.Scale(zero=False)),
                y=alt.Y("Environmental Impact:Q", title="Environmental Impact", scale=alt.Scale(zero=False)),
                color=alt.Color("Solution ID:N", legend=None),
                tooltip=["Solution ID", "NPV (Million €)", "Environmental Impact"]
            ).interactive()

            selection = alt.selection_point(
                name="solution_selector",
                fields=["Solution ID"],
                nearest=True,
                on="mouseover",
                empty="none"
            )

            points = scatter.add_selection(selection)

            # Add text labels for solution IDs
            text = alt.Chart(sorted_objective_data).mark_text(align='left', dx=7).encode(
                x="NPV (Million €):Q",
                y="Environmental Impact:Q",
                text="Solution ID:N",
                opacity=alt.condition(selection, alt.value(1), alt.value(0))
            )

            # Create the final chart
            final_chart = (points + text).properties(
                width=600,
                height=400,
                title="Click on a solution to view details"
            )

            col1, col2 = st.columns([2, 1])

            with col1:
                st.altair_chart(final_chart, use_container_width=True)

            with col2:
                st.subheader("Solution Explorer")

                # Sort solutions by ID in ascending order
                sorted_solution_ids = sorted(sorted_objective_data["Solution ID"].tolist())

                selected_solution_id = st.selectbox(
                    "Select a solution to view details:",
                    sorted_solution_ids,  # Use sorted list
                    index=0
                )

                # Find the selected solution
                selected_idx = sorted_objective_data[sorted_objective_data["Solution ID"] == selected_solution_id].index[0]
                selected_solution = sorted_allocation_data.iloc[selected_idx].iloc[:4].values
                selected_objectives = sorted_objective_data.iloc[selected_idx]

                st.write(f"**NPV:** {selected_objectives['NPV (Million €)']:.2f} Million €")
                st.write(f"**Environmental Impact:** {selected_objectives['Environmental Impact']:.4f}")

                # Store the selected solution
                st.session_state.selected_solution = selected_solution

            # Display the selected solution details
            st.subheader("Detailed Analysis of Selected Solution")
            if st.session_state.selected_solution is not None:
                col1, col2 = st.columns([1, 1])

                with col1:
                    # Show allocation as a pie chart
                    visualize_solution(
                        st.session_state.selected_solution,
                        total_waste,
                        solution_id=selected_solution_id
                    )

                with col2:
                    # Calculate benefits
                    selected_sol = st.session_state.selected_solution
                    x_inc, x_ad_chp, x_ad_up, x_comp = selected_sol

                    # Prepare parameters for TEA calculations
                    tea_params = {
                        'electricity_price': electricity_price,
                        'heat_price': heat_price,
                        'biomethane_tariff': biomethane_tariff,
                        'price_N': price_N,
                        'price_P': price_P,
                        'price_K': price_K,
                        'compost_per_ton': compost_per_ton
                    }
                    optimization_params = {'discount_rate': discount_rate, 'tax_rate': tax_rate}
                    params = {**tea_params, **optimization_params}

                    # Create problem to access calculation methods
                    problem = WasteTreatmentProblem(total_waste, params)

                    # Calculate NPV breakdown
                    npv_inc = problem._calculate_incineration_npv(x_inc) / NPV_SCALE
                    npv_ad_chp = problem._calculate_ad_chp_npv(x_ad_chp) / NPV_SCALE
                    npv_ad_up = problem._calculate_ad_upgrade_npv(x_ad_up) / NPV_SCALE
                    npv_comp = problem._calculate_compost_npv(x_comp) / NPV_SCALE

                    # Calculate environmental impact breakdown
                    env_inc = ENV_IMPACT['Incineration'] * x_inc
                    env_ad_chp = ENV_IMPACT['AD_CHP'] * x_ad_chp
                    env_ad_up = ENV_IMPACT['AD_Upgrade'] * x_ad_up
                    env_comp = ENV_IMPACT['Composting'] * x_comp

                    # Create data for Bar charts
                    npv_data = pd.DataFrame({
                        'Technology': ["Incineration", "AD with CHP", "AD with Upgrading", "Composting"],
                        'NPV (Million €)': [npv_inc, npv_ad_chp, npv_ad_up, npv_comp]
                    })

                    env_data = pd.DataFrame({
                        'Technology': ["Incineration", "AD with CHP", "AD with Upgrading", "Composting"],
                        'Environmental Impact': [env_inc, env_ad_chp, env_ad_up, env_comp]
                    })

                    # NPV breakdown chart
                    npv_chart = alt.Chart(npv_data).mark_bar().encode(
                        x=alt.X('Technology:N', title=None),
                        y=alt.Y('NPV (Million €):Q'),
                        color=alt.Color('Technology:N', scale=alt.Scale(scheme='category10')),
                        tooltip=['Technology', 'NPV (Million €)']
                    ).properties(
                        title='NPV Contribution by Technology',
                        height=250
                    )

                    st.altair_chart(npv_chart, use_container_width=True)

                    # Environmental impact breakdown chart
                    env_chart = alt.Chart(env_data).mark_bar().encode(
                        x=alt.X('Technology:N', title=None),
                        y=alt.Y('Environmental Impact:Q'),
                        color=alt.Color('Technology:N', scale=alt.Scale(scheme='category10')),
                        tooltip=['Technology', 'Environmental Impact']
                    ).properties(
                        title='Environmental Impact by Technology',
                        height=250
                    )

                    st.altair_chart(env_chart, use_container_width=True)

            # Compare solutions section
            st.subheader("Compare Solutions")

            # Allow comparison of multiple solutions
            if len(solutions) >= 3:
                st.write("Select solutions to compare:")
                
                col1, col2, col3 = st.columns(3)
                
                

                with col1:
                    compare_solution_id1 = st.selectbox(
                        "Solution 1:",
                        sorted_solution_ids,  # Use sorted list
                        index=0
                    )

                with col2:
                    # Default second solution is different from the first
                    default_index2 = 1 if len(sorted_solution_ids) > 1 else 0
                    compare_solution_id2 = st.selectbox(
                        "Solution 2:",
                        sorted_solution_ids,  # Use sorted list
                        index=default_index2
                    )

                with col3:
                    default_index3 = 2 if len(sorted_solution_ids) > 2 else default_index2
                    compare_solution_id3 = st.selectbox(
                        "Solution 3:",
                        sorted_solution_ids,  # Use sorted list
                        index=default_index3
                    )

                if len({compare_solution_id1, compare_solution_id2, compare_solution_id3}) == 3:
                    idx1 = sorted_objective_data[sorted_objective_data["Solution ID"] == compare_solution_id1].index[0]
                    idx2 = sorted_objective_data[sorted_objective_data["Solution ID"] == compare_solution_id2].index[0]
                    idx3 = sorted_objective_data[sorted_objective_data["Solution ID"] == compare_solution_id3].index[0]
                    
                    solution1 = sorted_allocation_data.iloc[idx1].iloc[:4].values
                    solution2 = sorted_allocation_data.iloc[idx2].iloc[:4].values
                    solution3 = sorted_allocation_data.iloc[idx3].iloc[:4].values
                    
                    objectives1 = sorted_objective_data.iloc[idx1]
                    objectives2 = sorted_objective_data.iloc[idx2]
                    objectives3 = sorted_objective_data.iloc[idx3]
                    
                    obj_comparison = pd.DataFrame({
                        'Metric': ['NPV (Million €)', 'Environmental Impact'],
                        f'Solution {compare_solution_id1}': [objectives1['NPV (Million €)'], objectives1['Environmental Impact']],
                        f'Solution {compare_solution_id2}': [objectives2['NPV (Million €)'], objectives2['Environmental Impact']],
                        f'Solution {compare_solution_id3}': [objectives3['NPV (Million €)'], objectives3['Environmental Impact']]
                    })
                    
                    st.write("### Objective Comparison")
                    st.dataframe(obj_comparison)
                    
                    alloc_comparison = pd.DataFrame({
                        'Technology': ["Incineration", "AD with CHP", "AD with Upgrading", "Composting"],
                        f'Solution {compare_solution_id1} (tons)': solution1,
                        f'Solution {compare_solution_id2} (tons)': solution2,
                        f'Solution {compare_solution_id3} (tons)': solution3,
                    })
                    
                    st.write("### Allocation Comparison")
                    st.dataframe(alloc_comparison)
                    
                    viz_data = pd.DataFrame({
                        'Technology': ["Incineration", "AD with CHP", "AD with Upgrading", "Composting"] * 3,
                        'Allocation (tons)': np.concatenate([solution1, solution2, solution3]),
                        'Solution': [f"Solution {compare_solution_id1}"] * 4 + [f"Solution {compare_solution_id2}"] * 4 + [f"Solution {compare_solution_id3}"] * 4
                    })
                    
                    bar_chart = alt.Chart(viz_data).mark_bar().encode(
                        x=alt.X('Technology:N', title='Technology'),
                        y=alt.Y('Allocation (tons):Q'),
                        color=alt.Color('Solution:N', scale=alt.Scale(scheme='set1')),
                        column=alt.Column('Solution:N', spacing=10),
                        tooltip=['Technology', 'Allocation (tons)', 'Solution']
                    ).properties(
                        title='Allocation Comparison by Technology',
                        height=350,
                        width=350
                    )
                    
                    st.altair_chart(bar_chart, use_container_width=False)
                else:
                    st.warning("Please select three different solutions to compare.")
            else:
                st.info("Need at least three solutions to enable comparison.")


            # Download results section
            st.subheader("Download Results")

            # Prepare full allocation data for download
            download_allocation = allocation_data.copy()
            download_allocation["NPV (Million €)"] = objective_data["NPV (Million €)"]
            download_allocation["Environmental Impact"] = objective_data["Environmental Impact"]

            def convert_df_to_csv(df):
                return df.to_csv().encode('utf-8')

            csv = convert_df_to_csv(download_allocation)
            st.download_button(
                label="Download Complete Results as CSV",
                data=csv,
                file_name='waste_treatment_optimization_results.csv',
                mime='text/csv',
            )

            # Export selected solution
            if st.session_state.selected_solution is not None:
                selected_sol_data = pd.DataFrame({
                    'Technology': ["Incineration", "AD with CHP", "AD with Upgrading", "Composting"],
                    'Allocation (tons)': st.session_state.selected_solution,
                    'Percentage (%)': (st.session_state.selected_solution / total_waste * 100).round(1)
                })

                selected_csv = convert_df_to_csv(selected_sol_data)
                st.download_button(
                    label="Download Selected Solution as CSV",
                    data=selected_csv,
                    file_name=f'solution_{selected_solution_id}_details.csv',
                    mime='text/csv',
                    key='download_selected'
                )
        else:
            st.info("No optimization results yet. Please run an optimization in the 'Decision Maker Preferences' tab.")

# Additional information and documentation for sidebar
def show_sidebar_info():
    st.sidebar.header("About this App")
    st.sidebar.write("""
    This interactive application helps decision-makers optimize waste treatment allocation across four technologies:

    - Incineration
    - Anaerobic Digestion with Combined Heat & Power (CHP)
    - Anaerobic Digestion with Biogas Upgrading
    - Composting

    The optimization considers both economic performance (NPV) and environmental impacts.
    """)

    st.sidebar.subheader("How to Use")
    st.sidebar.write("""
    1. Set your waste quantity and economic parameters
    2. Define your preferences and constraints
    3. Run the optimization
    4. Explore the Pareto-optimal solutions
    5. Compare different solutions
    6. Export your results
    """)

    st.sidebar.subheader("Methodology")
    st.sidebar.write("""
    The application uses:
    - NSGA-II algorithm for multi-objective optimization
    - Techno-economic assessment (TEA) models for each technology
    - Life cycle assessment (LCA) data for environmental impacts
    """)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2025 Waste Treatment Optimization")

# Main function
if __name__ == "__main__":
    import streamlit as st
    import numpy as np
    import pandas as pd
    import altair as alt
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.util.normalization import Normalization
    from src.core.economic.economic_analysis import (
        AnaerobicDigestionTEA,
        IncinerationTEA,
        CompostingTEA,
        NPVAnalysis
    )

    # Set page config - MUST BE THE FIRST STREAMLIT COMMAND
    st.set_page_config(page_title="Interactive Waste Treatment Optimization", layout="wide")

    # App title and description
    st.title("Interactive Waste Treatment Optimization")
    st.write("Optimize waste allocation across multiple treatment technologies considering economic and environmental objectives.")

    # Show sidebar information
    show_sidebar_info()

    # Run the main application
    run_optimization_analysis()
