"""Environmental Analysis Module

This module provides functionality for analyzing and visualizing environmental impacts
of different biowaste treatment technologies. It includes tools for comparing impacts
across multiple categories and decomposing impacts into their constituent parts.

The module uses Streamlit for the web interface and Plotly for interactive visualizations.
It supports analysis of three main impact categories:
- Human Health
- Ecosystem
- Resources

Features:
- Interactive selection of technologies and impact categories
- Detailed decomposition analysis of impacts
- Scientific notation for small values
- Comprehensive data tables with totals

Typical usage example:
    >>> import streamlit as st
    >>> from environmental_analysis import run_environmental_analysis
    >>> run_environmental_analysis()
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

# Organize decomposition data by impact categories
impact_categories = {
    'Human Health': [
        "Global Warming",
        "Ozone formation, human health",
        "Stratospheric ozone depletion",
        "Human carcinogenic toxicity",
        "Human non-carcinogenic toxicity"
    ],
    'Ecosystem': [
        "Terrestrial acidification",
        "Terrestrial ecotoxicity",
        "Ozone formation, Terrestrial ecosystems",
        "Global warming",
        "Freshwater eutrophication",
        "Freshwater ecotoxicity",
        "Marine ecotoxicity"
    ],
    'Resources': [
        "Fossil resource scarcity"
    ]
}

class EnvironmentalImpactCalculator:
    def __init__(self):
        # Impact factors for emissions (kg CO2-eq per kg emission)
        self.gwp_factors = {
            'CO2': 1.0,      # Global Warming Potential
            'CH4': 28.0,     # Methane GWP (100-year)
            'N2O': 265.0,    # Nitrous oxide GWP (100-year)
            'NH3': 0.0       # Ammonia (no direct GWP)
        }
        
        # Characterization factors for other impacts
        self.acidification_factors = {
            'NH3': 1.88,     # kg SO2-eq per kg
            'N2O': 0.7,      # kg SO2-eq per kg
        }
        
        self.eutrophication_factors = {
            'NH3': 0.35,     # kg PO4-eq per kg
            'N2O': 0.13,     # kg PO4-eq per kg
        }
        
    def calculate_impacts(self, emissions, energy_consumption=0, water_consumption=0):
        """Calculate environmental impacts from emissions and resource consumption
        
        Args:
            emissions: Dictionary of emissions in kg
            energy_consumption: Energy used in kWh
            water_consumption: Water used in m3
            
        Returns:
            Dictionary of impact scores
        """
        # Calculate Global Warming impact
        gwp = sum(emissions.get(gas, 0) * factor 
                  for gas, factor in self.gwp_factors.items())
                  
        # Calculate Acidification impact
        acidification = sum(emissions.get(gas, 0) * factor 
                          for gas, factor in self.acidification_factors.items())
                          
        # Calculate Eutrophication impact
        eutrophication = sum(emissions.get(gas, 0) * factor 
                           for gas, factor in self.eutrophication_factors.items())
                           
        # Add energy and water impacts
        resource_depletion = energy_consumption * 0.0067 + water_consumption * 0.001
        
        return {
            'Human Health': gwp * 1.4E-6 + acidification * 2.0E-6,  # Normalized scores
            'Ecosystem': eutrophication * 5.0E-6 + gwp * 8.0E-7,
            'Resources': resource_depletion
        }

# Initialize calculator
impact_calculator = EnvironmentalImpactCalculator()

# Data structure for results
data = {'Technology': [], 'Human Health': [], 'Ecosystem': [], 'Resources': []}
df = pd.DataFrame(data)

# Updated decomposition data structure to match the technologies
decomposition_data = {
    'Impact Category': [
        "Global Warming", "Ozone formation, human health", "Stratospheric ozone depletion",
        "Human carcinogenic toxicity", "Human non-carcinogenic toxicity",
        "Terrestrial acidification", "Terrestrial ecotoxicity",
        "Ozone formation, Terrestrial ecosystems", "Freshwater eutrophication",
        "Freshwater ecotoxicity", "Marine ecotoxicity", "Fossil resource scarcity"
    ],
    'Category': [
        "Human Health", "Human Health", "Human Health", "Human Health", "Human Health",
        "Ecosystem", "Ecosystem", "Ecosystem", "Ecosystem",
        "Ecosystem", "Ecosystem", "Resources"
    ],
    'Incineration - Biowaste': [0.06061, 3.46E-05, 1.59E-06, 0.014686, 0.284711,
                                3.25E-05, 0.108609, 3.63E-05, 2.7E-05,
                                0.008206, 0.011087, 0.005158],
    'Incineration - Energy Credit': [-0.47044, -0.00159, -2.5E-07, -0.00736, -0.07861,
                                    -0.00233, -1.39057, -0.00166, -1.8E-05,
                                    -0.0026, -0.00446, -0.14113],
    'Composting - Biowaste': [0.022779, 0.000256, 2.45E-08, 0.000683, 0.011062,
                             0.000126, 0.040623, 0.000258, 2.17E-06,
                             0.000196, 0.000294, 0.006806],
    'Composting - Compost Credit': [-0.00721, -8.4E-05, -7.4E-09, -0.00026, -0.00313,
                                   -3.9E-05, -0.01499, -8.4E-05, -4.7E-07,
                                   -8.3E-05, -0.00012, -0.00223],
    'AD - Biowaste': [0.105909, 2.57E-05, 6.93E-09, 0.001673, 0.016437,
                      4.22E-05, 0.046783, 2.71E-05, 4.64E-06,
                      0.001078, 0.001374, 0.003914],
    'AD - Energy Credit': [-0.43726, -0.00148, -2.3E-07, -0.00684, -0.07307,
                          -0.00216, -1.29251, -0.00154, -1.7E-05,
                          -0.00241, -0.00415, -0.13118],
    'AD - Digestate Credit': [-0.00604, -1.3E-05, -5.7E-09, -0.00025, -0.01091,
                             -2.8E-05, -0.02454, -1.3E-05, -2.6E-06,
                             -0.00024, -0.00035, -0.00243]
}

decomposition_df = pd.DataFrame(decomposition_data)

def create_main_plot(filtered_df, selected_impacts, selected_technologies):
    fig = go.Figure()

    bar_width = 0.2
    y_axis_ranges = {}  # Store y-axis ranges per category
    x_positions = {impact: [] for impact in selected_impacts}  # Tracking x positions for each impact

    # Calculate y-axis ranges for each impact
    for impact in selected_impacts:
        max_val = filtered_df[impact].max()
        min_val = filtered_df[impact].min()
        margin = (max_val - min_val) * 0.1 if max_val > min_val else 1e-6
        y_axis_ranges[impact] = (min(min_val - margin, 0), max_val + margin)

    # Add bars for each technology and impact category
    for i, impact in enumerate(selected_impacts):
        # Calculate the position of the bars to avoid overlap
        for j, tech in enumerate(selected_technologies):
            tech_data = filtered_df[filtered_df['Technology'] == tech]
            offset = j * bar_width  # offset by technology index

            fig.add_trace(go.Bar(
                name=f"{tech} - {impact}",
                x=[impact],
                y=tech_data[impact],
                text=[f"{val:.2e}" if abs(val) < 1e-3 else f"{val:.3f}" for val in tech_data[impact].values],
                textposition='auto',
                width=bar_width,
                offsetgroup=j,  # Offset bars based on technology index
            ))

    fig.update_layout(
        barmode='group',  # Group bars by impact category
        title="Environmental Single Scores Comparison",
        title_font=dict(size=30),
        xaxis_title="Impact Category",
        yaxis_title="Score",
        xaxis_title_font=dict(size=22),  # Set the font size of the x-axis title
        yaxis_title_font=dict(size=22),  # Set the font size of the y-axis title
        height=600,
        legend_title="Technologies",
        template="plotly_white",
        yaxis=dict(
            tickformat=".2e",  # Scientific notation for small numbers
            tickfont=dict(size=14),
            range=[min(y_axis_ranges[impact][0] for impact in selected_impacts),
                   max(y_axis_ranges[impact][1] for impact in selected_impacts)]
        ),
        xaxis=dict(
            tickvals=list(range(len(selected_impacts))),
            ticktext=selected_impacts,
            tickangle=0,  # Horizontal labels
            tickfont=dict(size=14)
        ),
        legend=dict(
            title=dict(font=dict(size=22)),  # Set the font size of the legend title
            font=dict(size=14),  # Set the font size of the legend labels
        ),
        margin=dict(b=50)
    )

    return fig

def create_decomposition_plot(decomposition_df, selected_technologies, selected_category):
    category_impacts = impact_categories[selected_category]
    filtered_df = decomposition_df[decomposition_df['Impact Category'].isin(category_impacts)]

    # Creating the columns dynamically based on selected technologies
    tech_columns = []
    if 'Incineration' in selected_technologies:
        tech_columns += ['Incineration - Biowaste', 'Incineration - Energy Credit']
    if 'Composting' in selected_technologies:
        tech_columns += ['Composting - Biowaste', 'Composting - Compost Credit']
    if 'AD' in selected_technologies:
        tech_columns += ['AD - Biowaste', 'AD - Digestate Credit', 'AD - Energy Credit']

    # Initialize the figure
    fig = go.Figure()

    bar_width = 0.2  # Set the width for each bar
    x_positions = range(len(filtered_df))  # The x positions for each category

    # Loop through each technology and add a bar trace to the figure
    for i, tech_col in enumerate(tech_columns):
        sign = 1 if 'Credit' in tech_col else 1  # Flip sign for credits if applicable
        fig.add_trace(go.Bar(
            name=tech_col,
            x=filtered_df['Impact Category'],
            y=filtered_df[tech_col] * sign,
            text=[f"{val:.2e}" if abs(val) < 1e-3 else f"{val:.4f}" for val in filtered_df[tech_col] * sign],
            textposition='inside',  # Position text inside the bars
            textangle=0,  # Explicitly set the text angle to 0 (horizontal)
            width=bar_width,
            offsetgroup=i,  # This makes bars of each technology grouped together
        ))

    fig.update_layout(
        title=f"Decomposition of {selected_category} Impacts",
        xaxis_title="Impact Category",
        yaxis_title="Score",
        height=600,
        barmode='group',  # Bars grouped by technology
        template="plotly_white",
        xaxis=dict(
            tickangle=45,  # Keep this as is to make x-axis labels readable
            tickfont=dict(size=12),
            title_font=dict(size=14)
        ),
        yaxis=dict(tickformat=".4f"),
        legend=dict(
            title=dict(font=dict(size=16)),
            font=dict(size=12),
        ),
        margin=dict(b=150)
    )

    return fig

def update_from_composting_model(composting_model):
    """Update environmental impacts using actual process data
    
    Args:
        composting_model: Instance of CompostingModel with simulation results
    """
    global df
    
    if hasattr(composting_model, 'cumulative_emissions'):
        # Calculate impacts using actual emissions
        impacts = impact_calculator.calculate_impacts(
            emissions=composting_model.cumulative_emissions,
            energy_consumption=composting_model.df_combined.get('energy_consumption', 0),
            water_consumption=composting_model.df_combined.get('water_consumption', 0)
        )
        
        # Update dataframe with calculated impacts
        new_row = {
            'Technology': 'Composting',
            'Human Health': impacts['Human Health'],
            'Ecosystem': impacts['Ecosystem'],
            'Resources': impacts['Resources']
        }
        
        # Update or append the composting row
        mask = df['Technology'] == 'Composting'
        if mask.any():
            df.loc[mask] = pd.Series(new_row)
        else:
            df = df.append(new_row, ignore_index=True)

def run_environmental_analysis():
    st.title("Biowaste Valorization LCA Visualization")
    st.write("Compare environmental impacts across different treatment technologies")

    # Controls for first plot on the main page
    st.subheader("Comparison of Technologies")
    col1, col2 = st.columns(2)
    with col1:
        selected_technologies = st.multiselect(
            "Select Technologies to Compare",
            options=df['Technology'].unique(),
            default=df['Technology'].unique()
        )
    with col2:
        selected_impacts = st.multiselect(
            "Select Impact Categories",
            options=['Human Health', 'Ecosystem', 'Resources'],
            default=['Human Health', 'Ecosystem', 'Resources']
        )

    # Check if 'selected_technologies' and 'selected_impacts' are selected
    if selected_technologies and selected_impacts:
        # Filter the dataframe based on the selected technologies
        filtered_df = df[df['Technology'].isin(selected_technologies)]

        # Apply scientific notation formatting to all values in the dataframe
        formatted_df = filtered_df[['Technology'] + selected_impacts].copy()

        # Convert all impact category columns to numeric type
        for col in selected_impacts:
            formatted_df[col] = pd.to_numeric(formatted_df[col], errors='coerce')

            # Apply scientific notation formatting
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x:.2e}" if abs(x) < 1E-4 else f"{x:.4f}"  # Format with scientific notation
            )

        # Calculate total sum for each impact category (before formatting)
        formatted_df['Total'] = formatted_df[selected_impacts].apply(
            lambda row: row.astype(float).sum(skipna=True), axis=1
        )

        # Apply scientific notation formatting to the 'Total' column
        formatted_df['Total'] = formatted_df['Total'].apply(
            lambda x: f"{x:.2e}" if abs(x) < 1E-4 else f"{x:.4f}"  # Format with scientific notation
        )

        # Main comparison plot
        main_fig = create_main_plot(filtered_df, selected_impacts, selected_technologies)
        st.plotly_chart(main_fig, use_container_width=True)

        # Display the formatted data table with the Total column
        st.subheader("Data Table")
        st.dataframe(formatted_df)

        # Decomposition section
        st.subheader("Decomposition of Impacts")
        col1, col2 = st.columns(2)
        with col1:
            technology_choice = st.selectbox("Select Technology", selected_technologies)
        with col2:
            selected_category = st.selectbox("Select Impact Category", list(impact_categories.keys()))

        # Create and display decomposition plot
        decomp_fig = create_decomposition_plot(decomposition_df, technology_choice, selected_category)
        st.plotly_chart(decomp_fig, use_container_width=True)

        # Decomposition data table with Total column
        st.subheader(f"Decomposition Data for {selected_category} - {technology_choice}")
        decomposition_filtered = decomposition_df[decomposition_df['Impact Category'].isin(impact_categories[selected_category])]

        # Apply scientific notation for decomposition table based on the same condition
        decomposition_table = decomposition_filtered[['Impact Category'] + [col for col in decomposition_filtered.columns if technology_choice in col]]

        # Convert all decomposition columns to numeric type before formatting
        for col in decomposition_table.columns[1:]:
            decomposition_table[col] = pd.to_numeric(decomposition_table[col], errors='coerce')

            # Apply scientific notation formatting
            decomposition_table[col] = decomposition_table[col].apply(
                lambda x: f"{x:.2e}" if abs(x) < 1E-4 else f"{x:.4f}"  # Format with scientific notation
            )

        # Calculate total sum for each row in the decomposition table (before formatting)
        decomposition_table['Total'] = decomposition_table.iloc[:, 1:].apply(
            lambda row: row.astype(float).sum(skipna=True), axis=1
        )

        # Apply scientific notation formatting to the 'Total' column
        decomposition_table['Total'] = decomposition_table['Total'].apply(
            lambda x: f"{x:.2e}" if abs(x) < 1E-4 else f"{x:.4f}"  # Format with scientific notation
        )

        # Display the decomposition data table with the Total column
        st.dataframe(decomposition_table)

    else:
        st.warning("Please select at least one technology and impact category.")
