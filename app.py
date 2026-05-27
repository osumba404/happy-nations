import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="PolicyPulse Simulation Engine",
    page_icon="",
    layout="wide"
)

# Custom styling for a clean portfolio look
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .metric-box { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title(" PolicyPulse: Dynamic Public Well-Being Simulator")
st.markdown("---")

# 2. Load Assets
@st.cache_data
def load_data():
    # Load the master dataset we merged. We will use 2019 as our current baseline year.
    full_df = pd.read_csv('cleaned_happiness_data.csv')
    latest_year = full_df['Year'].max()
    return full_df[full_df['Year'] == latest_year].copy()

try:
    df_current = load_data()
    model = joblib.load('happiness_model.pkl')
except FileNotFoundError:
    st.error(" Missing required files! Please make sure 'cleaned_happiness_data.csv' and 'happiness_model.pkl' are in this directory.")
    st.stop()

# 3. Sidebar - Country Selection & Sliders
st.sidebar.header(" Step 1: Choose Nation")
selected_country = st.sidebar.selectbox("Select a country to simulate:", sorted(df_current['Country'].unique()))

# Extract baseline values for the chosen country
baseline_data = df_current[df_current['Country'] == selected_country].iloc[0]

# Compute current baseline rank
df_current = df_current.sort_values(by='Happiness_Score', ascending=False).reset_index(drop=True)
baseline_rank = df_current[df_current['Country'] == selected_country].index[0] + 1

st.sidebar.header("🛠️ Step 2: Implement Policies")
st.sidebar.write("Modify baseline metrics to see outcomes.")

# Define interactive sliders centered at the country's actual real baseline values
sim_gdp = st.sidebar.slider("Economy (GDP per Capita)", 0.0, 2.0, float(baseline_data['GDP_per_Capita']), step=0.05)
sim_social = st.sidebar.slider("Social Support Systems", 0.0, 2.0, float(baseline_data['Social_Support']), step=0.05)
sim_health = st.sidebar.slider("Life Expectancy / Health", 0.0, 2.0, float(baseline_data['Life_Expectancy']), step=0.05)
sim_freedom = st.sidebar.slider("Personal Freedom", 0.0, 1.0, float(baseline_data['Freedom']), step=0.05)
sim_generosity = st.sidebar.slider("Community Generosity", 0.0, 1.0, float(baseline_data['Generosity']), step=0.05)
sim_corruption = st.sidebar.slider("Anti-Corruption / Trust", 0.0, 1.0, float(baseline_data['Corruption']), step=0.05)

# 4. Core Prediction and Rank Recalculation Engine
input_features = np.array([[sim_gdp, sim_social, sim_health, sim_freedom, sim_generosity, sim_corruption]])
predicted_score = model.predict(input_features)[0]

# Ensure the score stays within reasonable boundaries (0 to 10 scale)
predicted_score = max(0.0, min(10.0, predicted_score))

# Create a temporary dataframe to compute the updated global ranking positions
df_simulated = df_current.copy()
df_simulated.loc[df_simulated['Country'] == selected_country, 'Happiness_Score'] = predicted_score
df_simulated = df_simulated.sort_values(by='Happiness_Score', ascending=False).reset_index(drop=True)
predicted_rank = df_simulated[df_simulated['Country'] == selected_country].index[0] + 1

# Calculate changes relative to baseline
score_delta = predicted_score - baseline_data['Happiness_Score']
rank_delta = int(baseline_rank - predicted_rank)  # Positive means rank improved (e.g., 50 down to 40)

# 5. Main Screen Layout split into two layout columns
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader(f"Simulation Profile: {selected_country}")
    st.write("Live impact projections based on your policy configurations:")
    
    # Render KPI Cards
    st.metric(
        label="Simulated Happiness Score",
        value=f"{predicted_score:.3f}",
        delta=f"{score_delta:+.3f} vs Baseline"
    )
    
    # Note: For global ranks, a negative delta display means a better position closer to #1
    st.metric(
        label="Projected Global Rank",
        value=f"#{predicted_rank} out of {len(df_current)}",
        delta=f"{rank_delta:+} positions shifted",
        delta_color="normal" if rank_delta >= 0 else "inverse"
    )
    
    st.markdown("---")
    st.markdown("###  Policy Briefing")
    if score_delta > 0.4:
        st.success(" **Exceptional Intervention:** This balanced framework triggers systemic quality-of-life enhancements across demographics.")
    elif score_delta < -0.4:
        st.error(" **Systemic Strain:** Budget cuts or severe institutional decay to these pillars severely degrade civilian stability.")
    else:
        st.info(" **Marginal Adjustments:** The model projects steady, minor variations. Drastic systemic shift requires heavier structural investment.")

with col2:
    st.subheader("Pillar Impact Variance")
    st.write("Comparing the original baseline against your simulated shifts:")
    
    # Transform structural format for cleaner plotting with Plotly
    chart_df = pd.DataFrame({
        'Pillar': ['Economy', 'Social Support', 'Health', 'Freedom', 'Generosity', 'Trust (Corruption)'],
        'Real Baseline': [baseline_data['GDP_per_Capita'], baseline_data['Social_Support'], baseline_data['Life_Expectancy'], baseline_data['Freedom'], baseline_data['Generosity'], baseline_data['Corruption']],
        'Simulated Policy': [sim_gdp, sim_social, sim_health, sim_freedom, sim_generosity, sim_corruption]
    }).melt(id_vars='Pillar', var_name='Scenario', value_name='Weight value')
    
    fig = px.bar(
        chart_df, 
        x='Pillar', 
        y='Weight value', 
        color='Scenario', 
        barmode='group',
        color_discrete_sequence=['#94A3B8', '#3B82F6'],
        height=450
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)