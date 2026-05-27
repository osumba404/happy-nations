import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="PolicyPulse Simulation Engine",
    page_icon=":material/analytics:",
    layout="wide"
)


# 2. Load Assets
@st.cache_data
def load_data():
    full_df = pd.read_csv('cleaned_happiness_data.csv')
    latest_year = full_df['Year'].max()
    # Ensure rankings are sorted correctly
    full_df = full_df.sort_values(by='Happiness_Score', ascending=False).reset_index(drop=True)
    return full_df, latest_year

try:
    df_all, current_year = load_data()
    df_current = df_all[df_all['Year'] == current_year].copy()
    model = joblib.load('happiness_model.pkl')
except FileNotFoundError:
    st.error(":material/error: Missing required files! Please verify 'cleaned_happiness_data.csv' and 'happiness_model.pkl' exist in this directory.")
    st.stop()

# Header banner
st.title(":material/monitoring: PolicyPulse: Public Well-Being Simulator")
st.markdown(f"**Data Version:** World Happiness Report Framework (Baseline Year: {current_year})")
st.markdown("---")

# Create Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    ":material/tune: Policy Simulation Workspace", 
    ":material/leaderboard: Global Leaderboard", 
    ":material/database: Dataset Explorer"
])

# ==========================================
# TAB 1: SIMULATION WORKSPACE
# ==========================================
with tab1:
    # Sidebar elements for Controls
    st.sidebar.header(":material/public: Step 1: Target Nation")
    selected_country = st.sidebar.selectbox("Select country to modify:", sorted(df_current['Country'].unique()))

    baseline_data = df_current[df_current['Country'] == selected_country].iloc[0]
    
    # Calculate initial baseline rank
    df_current_sorted = df_current.sort_values(by='Happiness_Score', ascending=False).reset_index(drop=True)
    baseline_rank = df_current_sorted[df_current_sorted['Country'] == selected_country].index[0] + 1

    st.sidebar.header(":material/tune: Step 2: Policy Directives")
    st.sidebar.write("Slide to simulate legal, social, or economic reforms.")

    sim_gdp = st.sidebar.slider("Economy (GDP per Capita)", 0.0, 2.5, float(baseline_data['GDP_per_Capita']), step=0.02)
    sim_social = st.sidebar.slider("Social Support Systems", 0.0, 2.5, float(baseline_data['Social_Support']), step=0.02)
    sim_health = st.sidebar.slider("Life Expectancy / Health", 0.0, 2.5, float(baseline_data['Life_Expectancy']), step=0.02)
    sim_freedom = st.sidebar.slider("Personal Freedom", 0.0, 1.5, float(baseline_data['Freedom']), step=0.02)
    sim_generosity = st.sidebar.slider("Community Generosity", 0.0, 1.5, float(baseline_data['Generosity']), step=0.02)
    sim_corruption = st.sidebar.slider("Anti-Corruption / Trust", 0.0, 1.5, float(baseline_data['Corruption']), step=0.02)

    # Prediction Engine Matrix Pipeline
    input_features = pd.DataFrame([{
        'GDP_per_Capita': sim_gdp,
        'Social_Support': sim_social,
        'Life_Expectancy': sim_health,
        'Freedom': sim_freedom,
        'Generosity': sim_generosity,
        'Corruption': sim_corruption
    }])

    predicted_score = max(0.0, min(10.0, model.predict(input_features)[0]))

    # Dynamic Global Rank Calculation
    df_simulated = df_current_sorted.copy()
    df_simulated.loc[df_simulated['Country'] == selected_country, 'Happiness_Score'] = predicted_score
    df_simulated = df_simulated.sort_values(by='Happiness_Score', ascending=False).reset_index(drop=True)
    predicted_rank = df_simulated[df_simulated['Country'] == selected_country].index[0] + 1

    score_delta = predicted_score - baseline_data['Happiness_Score']
    rank_delta = int(baseline_rank - predicted_rank)

    # Main Grid Layout
    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        st.subheader(f":material/analytics: Simulation Dashboard — {selected_country}")
        
        # Grid of KPI Cards
        kpi1, kpi2 = st.columns(2)
        with kpi1:
            st.metric(
                label="Simulated Score",
                value=f"{predicted_score:.3f}",
                delta=f"{score_delta:+.3f} vs Actual"
            )
        with kpi2:
            st.metric(
                label="Projected Rank",
                value=f"#{predicted_rank}",
                delta=f"{rank_delta:+} spots shifted",
                delta_color="normal" if rank_delta >= 0 else "inverse"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### :material/assignment: Executive Policy Briefing")
        
        # Comprehensive Data-driven Insights Box
        with st.container(border=True):
            # 1. Broad Verdict
            if score_delta > 0.5:
                st.success("**🏆 Strategic Leap:** This structural framework drives monumental quality-of-life additions.")
            elif score_delta < -0.5:
                st.error("**⚠️ Systemic Instability:** Drastic structural reductions threaten national societal cohesion.")
            else:
                st.info("**📈 Marginal Variance:** Minor administrative changes yield baseline adjustments.")

            # 2. Economy vs Human Capital Balance Assessment
            if sim_gdp > baseline_data['GDP_per_Capita'] and sim_social < baseline_data['Social_Support']:
                st.warning("**⚖️ Growth Asymmetry:** Prioritizing GDP expansion while reducing safety nets introduces inequality risk.")
            elif sim_social > baseline_data['Social_Support'] and sim_corruption > baseline_data['Corruption']:
                st.info("**🛡️ Social Trust Synergy:** Pairing strong safety nets with anti-corruption measures optimizes investment efficiency.")

            # 3. Structural Vector Spot-Checks
            if sim_corruption > 0.4:
                st.markdown("- **Institutional Integrity:** High public transparency functions as a force-multiplier for infrastructure development.")
            if sim_freedom < 0.2:
                st.markdown("- **Civil Friction:** Restrictive individual choices create talent-drain risks over extended durations.")
            if sim_health > 1.0:
                st.markdown("- **Demographic Bonus:** Strong health metrics lower state healthcare dependencies.")
            if sim_generosity < 0.1:
                st.markdown("- **Social Fragility:** Weak community generosity indicates reliance on state-backed programs.")

    with col2:
        st.subheader(":material/bar_chart: Structural Profile Shift")
        
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
            color_discrete_sequence=['#94a3b8', '#2563eb'],
            height=450
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: GLOBAL LEADERBOARD
# ==========================================
with tab2:
    st.subheader(f":material/leaderboard: World Rankings Baseline ({current_year})")
    st.write("Current official positions sorted by baseline index scores.")
    
    # Process dataframe presentation properties
    leaderboard_df = df_current_sorted.copy()
    leaderboard_df['Rank'] = leaderboard_df.index + 1
    
    # Re-order variables for natural scanning
    display_cols = ['Rank', 'Country', 'Happiness_Score', 'GDP_per_Capita', 'Social_Support', 'Life_Expectancy', 'Freedom', 'Corruption']
    leaderboard_df = leaderboard_df[display_cols]
    
    # Visual top-performers distribution chart
    top_n = st.slider("Select display density range:", 5, 50, 15)
    
    fig_rank = px.bar(
        leaderboard_df.head(top_n),
        x='Country',
        y='Happiness_Score',
        color='Happiness_Score',
        color_continuous_scale='Blues',
        text_auto='.2f',
        title=f"Top {top_n} Nations Index Distribution Overview"
    )
    fig_rank.update_layout(plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False)
    st.plotly_chart(fig_rank, use_container_width=True)
    
    # Render Interactive UI Dataframe Table
    st.dataframe(
        leaderboard_df,
        column_config={
            "Happiness_Score": st.column_config.NumberColumn("Happiness Index", format="%.3f"),
            "GDP_per_Capita": st.column_config.NumberColumn("GDP Factor", format="%.3f"),
            "Social_Support": st.column_config.NumberColumn("Social Support", format="%.3f"),
            "Life_Expectancy": st.column_config.NumberColumn("Health Factor", format="%.3f"),
        },
        hide_index=True,
        use_container_width=True
    )

# ==========================================
# TAB 3: DATASET EXPLORER
# ==========================================
with tab3:
    st.subheader(":material/folder_open: Deep Longitudinal Engine Registry")
    st.write("Access raw values consolidated across historical collection vectors (2015-2019).")
    
    # Filtering matrix
    filter_years = st.multiselect("Query Filter Target Years:", options=sorted(df_all['Year'].unique()), default=sorted(df_all['Year'].unique()))
    search_query = st.text_input("Query Filter Country Name Search:", "").strip()
    
    filtered_master = df_all[df_all['Year'].isin(filter_years)]
    if search_query:
        filtered_master = filtered_master[filtered_master['Country'].str.contains(search_query, case=False, na=False)]
        
    st.markdown(f"**Discovered Dimensions:** {filtered_master.shape[0]} Rows × {filtered_master.shape[1]} Columns")
    
    # Action download button pipeline
    csv_bytes = filtered_master.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=":material/download: Export Current Filter Context as CSV",
        data=csv_bytes,
        file_name="policypulse_filtered_export.csv",
        mime="text/csv"
    )
    
    st.dataframe(filtered_master, use_container_width=True)