import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import plotly.express as px
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go
from scipy.stats import pearsonr

# --- Page Configuration ---
st.set_page_config(
    page_title="Air Quality Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- App Initialization & Loading Animation ---
progress_bar = st.progress(0, text="Initializing dashboard components...")
for i in range(100):
    time.sleep(0.01)
    progress_bar.progress(i + 1, text="Loading...")
time.sleep(0.5)
progress_bar.empty()
st.success('Dashboard loaded successfully!')

# --- Helper Functions with caching ---
@st.cache_resource
def load_trained_model():
    """Loads the pre-trained model."""
    MODEL_PATH = 'models/trained_model.pkl'
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Error: Model not found at '{MODEL_PATH}'. Please run 'process_and_train.py' first.")
        return None

@st.cache_data
def load_raw_data():
    """Loads and preprocesses the raw data for visualizations."""
    DATA_PATH = 'data/city_day.csv'
    try:
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error(f"Error: Data not found at '{DATA_PATH}'. Please download the dataset.")
        return None

# --- Main App ---

# --- Landing Page Section ---
st.container()
st.title("🌬️ Air Quality Forecasting & Analysis")
st.markdown(
    """
    **A machine learning dashboard to predict and visualize air quality trends.**
    
    Explore historical data and get a real-time prediction of future PM2.5 levels.
    """
)
st.markdown("---")

# --- Sidebar Menu ---
st.sidebar.header("Dashboard Controls")
st.sidebar.markdown("Use the toggles below to show or hide different sections.")

# Toggle buttons for each section
show_prediction = st.sidebar.toggle("Show PM2.5 Prediction", value=False)
show_data_vis = st.sidebar.toggle("Show Data Visualizations", value=False)
show_kpis = st.sidebar.toggle("Show Key Metrics", value=False)
show_data_table = st.sidebar.toggle("Show Raw Dataset Table", value=False)

# --- Prediction Section (Toggled) ---
if show_prediction:
    model = load_trained_model()
    if model is not None:
        st.header("🔮 Get a PM2.5 Prediction")
        st.markdown("Select a date to forecast the PM2.5 air quality level.")

        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                input_date = st.date_input("Select Date", datetime.now().date())
            with col2:
                # Placeholder for consistent layout
                st.empty() 

            if st.button("Predict PM2.5"):
                try:
                    # Create a DataFrame with the same features as the training data
                    input_df = pd.DataFrame([{'Date': pd.to_datetime(input_date)}])
                    input_df['dayofweek'] = input_df['Date'].dt.dayofweek
                    input_df['dayofyear'] = input_df['Date'].dt.dayofyear
                    input_df['month'] = input_df['Date'].dt.month
                    input_df['year'] = input_df['Date'].dt.year
                    input_df['is_weekend'] = input_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
                    
                    features = ['dayofweek', 'dayofyear', 'month', 'year', 'is_weekend']
                    prediction = model.predict(input_df[features])
                    
                    st.success(f"Predicted PM2.5 Level: **{prediction[0]:.2f}**")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
    st.markdown("---")

# --- Data Table Section (Toggled) ---
if show_data_table:
    raw_df = load_raw_data()
    if raw_df is not None:
        st.header("Raw Data Table")
        st.markdown("This table provides a preview of the raw dataset.")
        st.dataframe(raw_df.head(20).style.highlight_max(axis=0), use_container_width=True)
    st.markdown("---")


# --- Visualization & Insights Section (Toggled) ---
if show_data_vis:
    raw_df = load_raw_data()
    if raw_df is not None:
        st.header("📈 Data Visualization & Insights")
        st.markdown("Explore the trends and seasonality in the historical air quality data.")

        # Data selection for visualization
        selected_city = st.selectbox('Select a City', raw_df['City'].unique())
        city_df = raw_df[raw_df['City'] == selected_city].copy()
        city_df.set_index('Date', inplace=True)
        
        # Add a multiselect for other pollutants
        pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO', 'NH3']
        selected_pollutants = st.multiselect("Select Pollutants to Visualize", options=pollutants, default=['PM2.5'])
        
        # Plot selected pollutants over time
        if selected_pollutants:
            st.subheader(f"Pollutant Trends for {selected_city}")
            fig_multi_trend = px.line(city_df, x=city_df.index, y=selected_pollutants, title=f"Pollutant Levels Over Time")
            fig_multi_trend.update_layout(
                plot_bgcolor='#f0f2f6',
                paper_bgcolor='white',
                margin=dict(t=50, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_multi_trend, use_container_width=True)

        col_vis1, col_vis2 = st.columns(2)
        with col_vis1:
            # Bar chart for average PM2.5 across all cities
            city_avg_df = raw_df.groupby('City')['PM2.5'].mean().reset_index().sort_values(by='PM2.5', ascending=False)
            fig_bar = px.bar(city_avg_df, x='City', y='PM2.5', title='Average PM2.5 by City')
            fig_bar.update_layout(
                plot_bgcolor='#f0f2f6',
                paper_bgcolor='white',
                margin=dict(t=50, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_vis2:
            # Correlation Heatmap
            st.subheader("Pollutant Correlation Heatmap")
            st.markdown("This heatmap shows the correlation between different pollutants.")
            
            # Select only numeric columns for correlation calculation to avoid ValueError
            numeric_cols = raw_df.select_dtypes(include=np.number).columns
            corr_df = raw_df[numeric_cols].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.columns,
                colorscale='Viridis',
                zmin=-1, zmax=1
            ))
            fig_corr.update_layout(
                title="Correlation Matrix of Pollutants",
                xaxis_title="",
                yaxis_title="",
                plot_bgcolor='#f0f2f6',
                paper_bgcolor='white',
                margin=dict(t=50, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Seasonal Trends")
        col_vis3, col_vis4 = st.columns(2)
        with col_vis3:
            # Monthly trend
            monthly_avg = city_df.groupby(city_df.index.month)['PM2.5'].mean().reset_index()
            monthly_avg['Date'] = pd.to_datetime(monthly_avg['Date'], format='%m').dt.strftime('%B')
            fig_monthly = px.line(monthly_avg, x='Date', y='PM2.5', title="Monthly Average PM2.5")
            fig_monthly.update_layout(
                plot_bgcolor='#f0f2f6',
                paper_bgcolor='white',
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

        with col_vis4:
            # Day of week trend
            day_of_week_avg = city_df.groupby(city_df.index.dayofweek)['PM2.5'].mean().reset_index()
            day_of_week_avg['Day'] = day_of_week_avg['Date'].map({0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'})
            fig_day = px.line(day_of_week_avg, x='Day', y='PM2.5', title="Day of Week Average PM2.5")
            fig_day.update_layout(
                plot_bgcolor='#f0f2f6',
                paper_bgcolor='white',
            )
            st.plotly_chart(fig_day, use_container_width=True)
        st.markdown("---")

# --- KPI Metrics Section (Toggled) ---
if show_kpis:
    raw_df = load_raw_data()
    if raw_df is not None:
        st.header("📊 Key Air Quality Metrics")
        st.markdown("Get a quick overview of key metrics from the dataset.")

        avg_pm25 = raw_df['PM2.5'].mean()
        max_pm25 = raw_df['PM2.5'].max()
        min_pm25 = raw_df['PM2.5'].min()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Average PM25", value=f"{avg_pm25:.2f}", delta="Dataset Wide")
        with col2:
            st.metric(label="Highest Recorded PM25", value=f"{max_pm25:.2f}", delta="Dataset Wide")
        with col3:
            st.metric(label="Lowest Recorded PM25", value=f"{min_pm25:.2f}", delta="Dataset Wide")
        st.markdown("---")
