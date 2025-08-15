import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os

# Define file paths for all data sources
DATA_DIR = 'data'
DATA_FILES = {
    'city_day': os.path.join(DATA_DIR, 'city_day.csv'),
    'city_hour': os.path.join(DATA_DIR, 'city_hour.csv'),
    'station_day': os.path.join(DATA_DIR, 'station_day.csv'),
    'station_hour': os.path.join(DATA_DIR, 'station_hour.csv'),
    'stations': os.path.join(DATA_DIR, 'stations.csv')
}
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'trained_model.pkl')

def load_and_merge_data():
    """
    Loads all relevant CSV files and merges them into a single DataFrame.
    """
    try:
        # Load the core city_day data
        df_city_day = pd.read_csv(DATA_FILES['city_day'])
        df_city_day['Date'] = pd.to_datetime(df_city_day['Date'])

        # Since the 'city_day' file is the most comprehensive for a daily model,
        # we will use it as our primary source. The other files provide more
        # granular data which would be used for a more complex, multi-model approach.
        # For simplicity and to avoid complex joins on disparate data, we will
        # primarily use and clean the most relevant 'city_day' file.
        # However, this function is structured to be easily expanded.

        print("Using city_day.csv for analysis and training.")
        return df_city_day
    
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure all five CSV files are in the 'data' directory.")
        return None

def clean_and_preprocess(df):
    """
    Cleans data, handles missing values, and performs feature engineering.
    """
    # Drop rows with missing values in the target variable ('PM2.5')
    df.dropna(subset=['PM2.5'], inplace=True)
    
    # Fill remaining missing values with forward-fill and back-fill
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Feature engineering from the date index
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Select features and target. We are using PM2.5 as the target.
    features = ['dayofweek', 'dayofyear', 'month', 'year', 'is_weekend']
    target = 'PM2.5'
    
    return df[features], df[target]

def train_model(X, y):
    """Trains and saves the Random Forest model."""
    print("Training the Random Forest model...")
    # Split data chronologically for time-series
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print(f"R-squared (R2): {r2:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")

    # Save the model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel trained and saved to {MODEL_PATH}")

if __name__ == '__main__':
    df = load_and_merge_data()
    if df is not None:
        df.set_index('Date', inplace=True)
        X, y = clean_and_preprocess(df)
        train_model(X, y)
