🌬️ Air Quality Forecasting & Analysis Dashboard
This project is a web-based dashboard built with Streamlit that provides a comprehensive view of historical air quality data and a machine learning model for future PM2.5 level predictions. The dashboard allows users to interactively explore data visualizations, view key metrics, and get a forecast for a specific date.

✨ Features
Interactive Data Visualizations: Explore pollutant trends over time, seasonal patterns (monthly and day of the week), and a correlation heatmap to understand the relationships between different pollutants.

PM2.5 Forecasting: Use the built-in machine learning model to predict the PM2.5 air quality level for any future date you select.

Key Metrics: Get a quick overview of important air quality statistics, including average, highest, and lowest recorded PM2.5 levels across the dataset.

Dynamic UI: Use the sidebar toggles to show or hide different sections of the dashboard, providing a clean and customizable user experience.

🚀 How to Run the App
Prerequisites
Python 3.x

The following Python libraries: streamlit, pandas, numpy, joblib, plotly, scikit-learn, scipy

Step-by-Step Instructions
Clone the repository or download the project files.

Ensure you have the necessary data and model files. This project assumes you have a data folder with city_day.csv and a models folder with trained_model.pkl.

Install the required libraries by running the following command in your terminal:

pip install streamlit pandas numpy joblib plotly scikit-learn scipy

Run the Streamlit app from your terminal in the same directory as the app.py file:

streamlit run app.py

Your web browser will automatically open to the dashboard.