import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# For data processing and GPR modeling
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, ConstantKernel as C
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.dates as mdates
import os

# For LSTM modeling using TensorFlow instead of PyTorch
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Set page config
st.set_page_config(
    page_title="Gaussian Process Regression Prediction - GPRO",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for improved visuals with gradient backgrounds
def add_custom_css():
    st.markdown("""
    <style>
        /* Main background and text colors */
        .main {
            background-color: #f5f5f5;
            color: #333333;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #e6e6e6;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #4f4f4f;
        }
        
        /* Cards and containers */
        .stCard, div.block-container {
            border-radius: 15px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            background-color: #a3c9c7;
            border: none;
            color: white;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 10px;
        }
        
        .stButton>button:hover {
            background-color: #8ab5b2;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        /* Metrics containers */
        .metric-container {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 10px;
        }
        
        /* Custom cards for dashboard with unique colors */
        .dashboard-card {
            border-radius: 15px;
            padding: 20px;
            margin: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: transform 0.3s ease;
            height: 100%;
            cursor: pointer;
        }
        
        .dashboard-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        }
        
        /* Card 1 - Prediction */
        .predict-card {
            background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
            color: #4f4f4f;
        }
        
        /* Card 2 - Historical Data */
        .historical-card {
            background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
            color: #4f4f4f;
        }
        
        /* Card 3 - OOS Evaluation */
        .oos-card {
            background: linear-gradient(135deg, #9795f0 0%, #fbc8d4 100%);
            color: #4f4f4f;
        }
        
        /* Card 4 - About */
        .about-card {
            background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
            color: #4f4f4f;
        }
        
        .card-icon {
            font-size: 50px;
            margin-bottom: 15px;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #777777;
        }
        
        /* Custom colors for specific elements */
        .highlight-text {
            color: #a3c9c7;
            font-weight: bold;
        }
        
        /* Make dashboard cards bigger */
        .big-dashboard-container {
            height: 400px;
            margin-top: 30px;
        }
        
        .big-dashboard-card {
            height: 350px; 
            padding: 30px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .big-card-icon {
            font-size: 80px;
            margin-bottom: 25px;
        }
        
        .big-card-title {
            font-size: 24px;
            margin-bottom: 15px;
        }
        
        .big-card-desc {
            font-size: 16px;
        }
        
        /* Navigation menu styling */
        .nav-item {
            background-color: #f2f2f2;
            margin-bottom: 8px;
            border-radius: 10px;
            overflow: hidden;
            transition: background-color 0.3s ease;
        }
        
        .nav-item:hover {
            background-color: #e6e6e6;
        }
        
        .nav-item-active {
            background-color: #a3c9c7;
            color: white;
        }
        
        .nav-link {
            display: block;
            padding: 12px 15px;
            text-decoration: none;
            color: #4f4f4f;
        }
        
        .nav-link-active {
            color: white;
            font-weight: bold;
        }
        
        .nav-item-icon {
            margin-right: 10px;
        }
        
        /* Download link styling */
        .download-link {
            display: inline-block;
            margin-top: 10px;
            padding: 8px 16px;
            background-color: #a3c9c7;
            color: white;
            border-radius: 5px;
            text-decoration: none;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .download-link:hover {
            background-color: #8ab5b2;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Model selection tabs */
        .model-tabs {
            display: flex;
            margin-bottom: 20px;
        }
        
        .model-tab {
            padding: 10px 20px;
            margin-right: 10px;
            border-radius: 5px;
            background-color: #f0f0f0;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .model-tab:hover {
            background-color: #e0e0e0;
        }
        
        .model-tab.active {
            background-color: #a3c9c7;
            color: white;
        }
        
        /* LSTM specific styles */
        .lstm-colors .stButton>button {
            background-color: #a3b5c9;
        }
        
        .lstm-colors .stButton>button:hover {
            background-color: #8aa4b5;
        }
    </style>
    """, unsafe_allow_html=True)

# Apply custom CSS
add_custom_css()

# Create a custom sidebar with styled navigation
def styled_sidebar(current_page):
    st.sidebar.image("https://img.icons8.com/pastel-glyph/64/000000/stocks-growth--v2.png", width=50)
    st.sidebar.title("Gaussian Process Regression Prediction - GPRO")
    st.sidebar.markdown("### by Raja Valentino Kristananda")
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.markdown("<h3>Navigation</h3>", unsafe_allow_html=True)
    
    # Define navigation items with icons
    nav_items = [
        {"name": "Home", "icon": "🏠"},
        {"name": "Predict Stock Prices", "icon": "🔮"},
        {"name": "Out-of-Sample Evaluation", "icon": "🔍"},
        {"name": "Historical Data", "icon": "📊"},
        {"name": "About", "icon": "ℹ️"}
    ]
    
    # Add custom CSS for navigation styling
    st.markdown("""
    <style>
    .nav-button {
        width: 100%;
        text-align: left;
        padding: 10px 15px;
        margin-bottom: 8px;
        border-radius: 10px;
        transition: background-color 0.3s;
        background-color: #f2f2f2;
        border: none;
        color: #4f4f4f;
    }
    
    .nav-button:hover {
        background-color: #e6e6e6;
    }
    
    .nav-button-active {
        background-color: #a3c9c7;
        color: white;
        font-weight: bold;
    }
    
    .nav-icon {
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    selected_page = current_page
    
    # Create buttons for each navigation item
    for item in nav_items:
        is_active = item["name"] == current_page
        btn_class = "nav-button nav-button-active" if is_active else "nav-button"
        
        if st.sidebar.button(
            f"{item['icon']} {item['name']}", 
            key=f"nav_{item['name']}",
            help=f"Navigate to {item['name']}"
        ):
            selected_page = item["name"]
    
    st.sidebar.markdown("---")
    return selected_page

# Function to create a dashboard card
def dashboard_card(title, description, icon, key, card_class=""):
    st.markdown(f"""
    <div class="dashboard-card {card_class}" onclick="document.getElementById('{key}').click()">
        <div class="card-icon">{icon}</div>
        <h3>{title}</h3>
        <p>{description}</p>
    </div>
    <button id="{key}" style="display:none;"></button>
    """, unsafe_allow_html=True)

# Functions for data processing and model training
def preprocess_data(df, stock_name):
    """
    Preprocess stock data for model training
    """
    # Create a copy of the dataframe
    data = df.copy()
    
    # Convert date column to datetime
    try:
        # Try DD/MM/YYYY format first
        data['Tanggal'] = pd.to_datetime(data['Tanggal'], dayfirst=True)
    except ValueError:
        try:
            # Try MM/DD/YYYY format
            data['Tanggal'] = pd.to_datetime(data['Tanggal'], dayfirst=False)
        except ValueError:
            # Let pandas guess the format
            data['Tanggal'] = pd.to_datetime(data['Tanggal'], errors='coerce')
    
    # Convert to numeric if not already - tanpa perkalian 1000
    if 'Terakhir' in data.columns:
        data['Terakhir'] = pd.to_numeric(data['Terakhir'], errors='coerce')
    
    # Sort by date (oldest to newest)
    data = data.sort_values('Tanggal')
    
    # Reset index
    data = data.reset_index(drop=True)
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        st.warning(f"Missing values detected: {missing_values}")
        data = data.fillna(method='ffill')
        st.info("Missing values handled using forward fill")
    
    # Add time features
    data['Year'] = data['Tanggal'].dt.year
    data['Month'] = data['Tanggal'].dt.month
    data['Day'] = data['Tanggal'].dt.day
    data['DayOfWeek'] = data['Tanggal'].dt.dayofweek
    
    # Determine optimal window sizes for moving averages based on dataset size
    data_size = len(data)
    window_size1 = min(5, max(2, data_size // 5))
    window_size2 = min(20, max(3, data_size // 3))
    
    # Add moving averages
    data[f'MA{window_size1}'] = data['Terakhir'].rolling(window=window_size1).mean()
    data[f'MA{window_size2}'] = data['Terakhir'].rolling(window=window_size2).mean()
    
    # Fill NaN values for small datasets
    data = data.fillna(method='bfill').fillna(method='ffill')
    
    # Add stock name
    data['Stock'] = stock_name
    
    return data, window_size1, window_size2

# GPR Functions
def prepare_gpr_features(data, window_size1, window_size2):
    """
    Prepare features for GPR model training
    """
    features = ['Year', 'Month', 'Day', 'DayOfWeek', f'MA{window_size1}', f'MA{window_size2}']
    X = data[features]
    y = data['Terakhir']
    
    # Scale features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, scaler_X, scaler_y, features

# LSTM Functions
def create_sequences(data, n_steps):
    """
    Create sequences for LSTM model
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps), :])
        y.append(data[i + n_steps, 0])  # Target is the price at the next time step
    return np.array(X), np.array(y)

def prepare_lstm_features(data, sequence_length=30):
    """
    Prepare features for LSTM model training
    """
    # Define features
    ma_cols = [col for col in data.columns if col.startswith('MA')]
    features = ['Terakhir'] + ma_cols + ['Year', 'Month', 'Day', 'DayOfWeek']
    
    # Ensure all features exist in dataframe
    features = [f for f in features if f in data.columns]
    
    # Extract raw data
    X_raw = data[features].values
    
    # Scale features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(X_raw)
    scaler_y.fit(data[['Terakhir']].values)
    
    # Transform features
    X_scaled = scaler_X.transform(X_raw)
    
    # Create sequences for LSTM
    X_seq, y_seq = create_sequences(X_scaled, sequence_length)
    
    return X_seq, y_seq, scaler_X, scaler_y, sequence_length, features

def split_training_data(X, y, data_size, model_type="GPR"):
    """
    Split data into training, validation, and testing sets
    """
    if model_type == "GPR":
        # For small datasets, ensure there's enough data for training
        min_train_size = max(3, data_size // 2)
        
        # Calculate appropriate test ratio
        test_ratio = min(0.2, 1 - (min_train_size / data_size))
        
        if data_size <= 10:
            # For very small datasets
            train_ratio = max(0.7, min_train_size / data_size)
            val_test_ratio = (1 - train_ratio) / 2
            
            split_idx_train = int(data_size * train_ratio)
            split_idx_val = split_idx_train + int(data_size * val_test_ratio)
            
            X_train = X[:split_idx_train]
            y_train = y[:split_idx_train]
            
            X_val = X[split_idx_train:split_idx_val]
            y_val = y[split_idx_train:split_idx_val]
            
            X_test = X[split_idx_val:]
            y_test = y[split_idx_val:]
        else:
            # For larger datasets (8:1:1 split)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_ratio, random_state=42, shuffle=False)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)
        
        # Ensure each set has at least one sample
        if len(X_val) == 0:
            X_val = X_train[-1:].copy()
            y_val = y_train[-1:].copy()
            X_train = X_train[:-1]
            y_train = y_train[:-1]
        
        if len(X_test) == 0:
            X_test = X_val[-1:].copy()
            y_test = y_val[-1:].copy()
            X_val = X_val[:-1]
            y_val = y_val[:-1]
    else:  # LSTM
        # For LSTM, sequence data requires slightly different handling
        n_samples = len(X)
        train_size = int(0.8 * n_samples)
        val_size = int(0.1 * n_samples)
        
        # Split data
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_gpr_models(X_train, y_train):
    """
    Train Gaussian Process Regression models with different kernels
    """
    # Define kernels
    kernels = {
        'RBF': C(1.0) * RBF(length_scale=1.0),
        'Matern': C(1.0) * Matern(length_scale=1.0, nu=1.5),
        'RationalQuadratic': C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0),
        'Linear': C(1.0) * DotProduct() + C(1.0)
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Train models
    trained_models = {}
    training_times = {}
    
    for i, (name, kernel) in enumerate(kernels.items()):
        status_text.text(f"Training {name} kernel model...")
        
        # Create and train model
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.1,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=42
        )
        
        # Record training time
        import time
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        
        training_time = end_time - start_time
        trained_models[name] = model
        training_times[name] = training_time
        
        # Update progress
        progress_value = (i + 1) / len(kernels)
        progress_bar.progress(progress_value)
        status_text.text(f"Trained {name} kernel in {training_time:.2f} seconds")
    
    status_text.text("All models trained successfully!")
    
    return trained_models, training_times

def build_lstm_model(input_shape, units=50, dropout=0.2, num_layers=1):
    """
    Build a TensorFlow LSTM model
    """
    model = Sequential()
    
    # First LSTM layer
    if num_layers > 1:
        model.add(LSTM(units=units, 
                     return_sequences=True, 
                     input_shape=input_shape))
        model.add(Dropout(dropout))
    else:
        model.add(LSTM(units=units, 
                     return_sequences=False, 
                     input_shape=input_shape))
        model.add(Dropout(dropout))
    
    # Additional LSTM layers if num_layers > 1
    for i in range(1, num_layers):
        if i < num_layers - 1:
            model.add(LSTM(units=units, return_sequences=True))
            model.add(Dropout(dropout))
        else:
            model.add(LSTM(units=units, return_sequences=False))
            model.add(Dropout(dropout))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer=Adam(), loss='mse')
    
    return model

def train_lstm_models(X_train, y_train, X_val, y_val, input_shape):
    """
    Train LSTM models with different architectures using TensorFlow
    """
    # Define LSTM architecture configurations
    lstm_configs = {
        'LSTM_50': {'units': 50, 'dropout': 0.2, 'num_layers': 1},
        'LSTM_100': {'units': 100, 'dropout': 0.2, 'num_layers': 1},
        'LSTM_50_higher_dropout': {'units': 50, 'dropout': 0.5, 'num_layers': 1},
        'LSTM_stacked': {'units': 50, 'dropout': 0.2, 'num_layers': 2}
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Training parameters
    batch_size = 32
    epochs = 100
    patience = 10
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    
    # Train models
    trained_models = {}
    training_times = {}
    histories = {}
    
    for i, (name, config) in enumerate(lstm_configs.items()):
        status_text.text(f"Training {name} model...")
        
        # Create model
        model = build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            units=config['units'],
            dropout=config['dropout'],
            num_layers=config['num_layers']
        )
        
        # Record training time
        import time
        start_time = time.time()
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Store model and metrics
        trained_models[name] = model
        training_times[name] = training_time
        histories[name] = history
        
        # Update progress
        progress_value = (i + 1) / len(lstm_configs)
        progress_bar.progress(progress_value)
        status_text.text(f"Trained {name} model in {training_time:.2f} seconds ({len(history.history['loss'])} epochs)")
    
    status_text.text("All models trained successfully!")
    
    return trained_models, training_times, histories

def calculate_metrics(y_true, y_pred, scaler_y, model_type="GPR"):
    """
    Calculate evaluation metrics for model performance
    """
    if model_type == "GPR":
        # Return to original scale for metrics
        y_true_orig = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:  # LSTM
        # For LSTM, we need to reshape for inverse transform
        y_true_reshaped = y_true.reshape(-1, 1)
        y_pred_reshaped = y_pred.reshape(-1, 1)
        
        # Return to original scale
        y_true_orig = scaler_y.inverse_transform(y_true_reshaped).flatten()
        y_pred_orig = scaler_y.inverse_transform(y_pred_reshaped).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    mse = mean_squared_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_orig, y_pred_orig)
    
    # Calculate MAPE (handle case when y_true = 0)
    epsilon = 1e-10  # Prevent division by zero
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + epsilon))) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def evaluate_models(trained_models, X_val, y_val, X_test, y_test, scaler_y, model_type="GPR", histories=None):
    """
    Evaluate models and return performance metrics
    """
    # Evaluate models on validation set
    val_results = {}
    
    for name, model in trained_models.items():
        if model_type == "GPR":
            y_val_pred = model.predict(X_val)
        else:  # LSTM with TensorFlow
            y_val_pred = model.predict(X_val, verbose=0)
        
        val_results[name] = calculate_metrics(y_val, y_val_pred, scaler_y, model_type)
    
    # Find best model based on RMSE
    best_model_name = min(val_results, key=lambda k: val_results[k]['RMSE'])
    best_model = trained_models[best_model_name]
    
    # Evaluate best model on test set
    if model_type == "GPR":
        y_test_pred = best_model.predict(X_test)
    else:  # LSTM with TensorFlow
        y_test_pred = best_model.predict(X_test, verbose=0)
    
    test_metrics = calculate_metrics(y_test, y_test_pred, scaler_y, model_type)
    
    # Calculate train metrics (for overfitting check)
    if model_type == "GPR":
        y_train_pred = best_model.predict(X_val)  # Using validation set for train metrics
    else:  # LSTM with TensorFlow
        y_train_pred = best_model.predict(X_val, verbose=0)
    
    train_metrics = calculate_metrics(y_val, y_train_pred, scaler_y, model_type)
    
    return best_model, best_model_name, val_results, test_metrics, train_metrics

def plot_stock_predictions(data, dates, y_full_orig, y_pred_full_orig, n_train, n_val, n_test, best_model_name, stock_name, model_type="GPR"):
    """
    Plot historical data with predictions
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual vs. predicted values
    ax.plot(dates, y_full_orig, label='Actual', color='#4286f4', linewidth=2, alpha=0.8)
    ax.plot(dates, y_pred_full_orig, label=f'Predicted ({best_model_name})', color='#ff8c00', linestyle='--', linewidth=2)
    
    # Highlight train/val/test regions
    ax.axvspan(dates[0], dates[n_train-1], alpha=0.2, color='green', label='Training Set')
    
    if n_val > 0:
        ax.axvspan(dates[n_train], dates[n_train+n_val-1], alpha=0.2, color='blue', label='Validation Set')
    
    if n_test > 0:
        ax.axvspan(dates[n_train+n_val], dates[-1], alpha=0.2, color='red', label='Testing Set')
    
    # Add labels and formatting
    model_name = 'Gaussian Process Regression' if model_type == 'GPR' else 'LSTM'
    ax.set_title(f'{stock_name} Stock Price Prediction with {model_name}', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price (Rp)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    fig.tight_layout()
    return fig

def plot_error_analysis(train_metrics, val_metrics, test_metrics):
    """
    Plot error metrics comparison
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE comparison
    ax1.bar(['Training', 'Validation', 'Testing'], 
            [train_metrics['MSE'], val_metrics['MSE'], test_metrics['MSE']],
            color=['green', 'blue', 'red'], alpha=0.7)
    ax1.set_title('Mean Squared Error (MSE)', fontsize=14)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add values to bars
    for i, v in enumerate([train_metrics['MSE'], val_metrics['MSE'], test_metrics['MSE']]):
        ax1.text(i, v + v*0.05, f'{v:.2f}', ha='center', va='bottom')
    
    # MAE comparison
    ax2.bar(['Training', 'Validation', 'Testing'], 
            [train_metrics['MAE'], val_metrics['MAE'], test_metrics['MAE']],
            color=['green', 'blue', 'red'], alpha=0.7)
    ax2.set_title('Mean Absolute Error (MAE)', fontsize=14)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add values to bars
    for i, v in enumerate([train_metrics['MAE'], val_metrics['MAE'], test_metrics['MAE']]):
        ax2.text(i, v + v*0.05, f'{v:.2f}', ha='center', va='bottom')
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.suptitle('Model Error Analysis', fontsize=16)
    
    return fig

def plot_lstm_training_history(histories, best_model_name):
    """
    Plot LSTM training history
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, history in histories.items():
        if name == best_model_name:
            ax.plot(history.history['loss'], label=f'Training Loss ({name})', linewidth=2)
            ax.plot(history.history['val_loss'], label=f'Validation Loss ({name})', linewidth=2)
        else:
            ax.plot(history.history['loss'], label=f'Training Loss ({name})', alpha=0.5)
            ax.plot(history.history['val_loss'], label=f'Validation Loss ({name})', alpha=0.5)
    
    ax.set_title('LSTM Training History', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss (MSE)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    return fig

def plot_future_predictions(historical_dates, historical_prices, future_dates, future_prices, stock_name, model_type="GPR"):
    """
    Plot historical data with future predictions
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    ax.plot(historical_dates, historical_prices, label='Historical Data', color='#4286f4', linewidth=2)
    
    # Plot future predictions
    ax.plot(future_dates, future_prices, label='Future Predictions', 
            color='#ff8c00', linestyle='--', linewidth=2, marker='o', markersize=5)
    
    # Add vertical line to mark prediction start
    ax.axvline(x=historical_dates[-1], color='gray', linestyle='--', alpha=0.7)
    
    # Add labels and formatting
    model_name = 'Gaussian Process Regression' if model_type == 'GPR' else 'LSTM'
    ax.set_title(f'Future Price Predictions for {stock_name} using {model_name}', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price (Rp)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    fig.tight_layout()
    return fig

def get_download_link(df, filename="download.csv"):
    """
    Create download link for dataframe
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">Download CSV File</a>'
    return href

def predict_lstm_future(best_model, sequence_length, data, scaler_X, scaler_y, features, future_days=30):
    """
    Predict future prices using LSTM model (TensorFlow version)
    """
    # Get last date from data
    last_date = data['Tanggal'].iloc[-1]
    
    # Generate future dates
    future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
    
    # Get the last sequence from the data for initial prediction
    feature_values = data[features].values
    last_sequence = feature_values[-sequence_length:]
    
    # Scale the sequence
    last_sequence_scaled = scaler_X.transform(last_sequence)
    
    # Historical prices for updating sequences
    historical_prices = data['Terakhir'].values.tolist()
    
    # Make predictions for future days
    predicted_prices = []
    
    for i in range(future_days):
        # Reshape for LSTM input (samples, timesteps, features)
        current_sequence = last_sequence_scaled.reshape(1, sequence_length, len(features))
        
        # Predict next price
        next_price_scaled = best_model.predict(current_sequence, verbose=0)[0][0]
        
        # Convert back to original scale
        next_price = scaler_y.inverse_transform(np.array([[next_price_scaled]]))[0][0]
        
        # Save prediction
        predicted_prices.append(next_price)
        
        # Prepare next day's features
        next_date = future_dates[i]
        new_row = []
        
        # Start with the predicted price
        new_row.append(next_price)
        
        # Add MA values
        ma_cols = [col for col in features if col.startswith('MA')]
        for col in ma_cols:
            window_size = int(col[2:])
            if i + 1 < window_size:
                # Use combination of historical and predicted data
                ma_value = np.mean(historical_prices[-(window_size-(i+1)):] + predicted_prices)
            else:
                # Use only predicted data
                ma_value = np.mean(predicted_prices[-window_size:])
            new_row.append(ma_value)
        
        # Add date features
        new_row.extend([
            next_date.year,
            next_date.month,
            next_date.day,
            next_date.weekday()
        ])
        
        # Create new scaled row
        new_row_array = np.array([new_row])
        new_row_scaled = scaler_X.transform(new_row_array)
        
        # Update the sequence for next prediction
        last_sequence_scaled = np.vstack((last_sequence_scaled[1:], new_row_scaled))
        
        # Update historical prices
        historical_prices.append(next_price)
    
    # Create DataFrame for predictions
    future_predictions = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': predicted_prices
    })
    
    return future_predictions

def predict_gpr_future(best_model, data, scaler_X, scaler_y, features, future_days=30):
    """
    Predict future prices using GPR model
    """
    # Get last date from data
    last_date = data['Tanggal'].iloc[-1]
    
    # Generate future dates
    future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
    
    # Create dataframe for prediction
    future_data = pd.DataFrame({'Tanggal': future_dates})
    future_data['Year'] = future_data['Tanggal'].dt.year
    future_data['Month'] = future_data['Tanggal'].dt.month
    future_data['Day'] = future_data['Tanggal'].dt.day
    future_data['DayOfWeek'] = future_data['Tanggal'].dt.dayofweek
    
    # Get MA window sizes from features
    ma_windows = []
    for feature in features:
        if feature.startswith('MA'):
            try:
                window_size = int(feature[2:])
                ma_windows.append(window_size)
            except ValueError:
                pass
    
    # Iterative prediction for each future day
    predicted_prices = []
    window_sizes = sorted(ma_windows)
    
    # Get historical prices for MA calculation
    historical_prices = data['Terakhir'].values.tolist()
    
    # Predict day by day
    for i in range(future_days):
        # Prepare row for prediction
        row_data = future_data.iloc[i:i+1].copy()
        
        # Calculate MA values
        for window_size in window_sizes:
            feature_name = f'MA{window_size}'
            
            if i < window_size:
                # Mix of historical and predicted prices
                prices_window = historical_prices[-window_size+i:] + predicted_prices
                row_data[feature_name] = np.mean(prices_window)
            else:
                # Only predicted prices
                row_data[feature_name] = np.mean(predicted_prices[-window_size:])
        
        # Make prediction
        X_future = row_data[features].values
        X_future_scaled = scaler_X.transform(X_future)
        y_future_scaled = best_model.predict(X_future_scaled)
        y_future = scaler_y.inverse_transform(y_future_scaled.reshape(-1, 1))[0][0]
        
        # Store prediction
        predicted_prices.append(y_future)
        
        # Update for next prediction
        historical_prices.append(y_future)
    
    # Create dataframe with predictions
    future_predictions = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': predicted_prices
    })
    
    return future_predictions

def predict_stock_prices(uploaded_file, stock_name, model_type="GPR"):
    """
    Main function to predict stock prices from uploaded file
    """
    try:
        # Load data
        if uploaded_file.name.lower().endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            # Gunakan parameter thousands='.' untuk menangani format angka Indonesia
            df = pd.read_csv(uploaded_file, thousands='.')
        
        # Check for required columns
        required_cols = ['Tanggal', 'Terakhir']
        if not all(col in df.columns for col in required_cols):
            st.error(f"File must have these columns: {required_cols}")
            return
        
        # Display data preview
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())
        
        # Preprocess data
        with st.spinner("Preprocessing data..."):
            processed_data, window_size1, window_size2 = preprocess_data(df, stock_name)
            
            # Data summary
            st.info(f"""
            **Data Summary:**
            - Total rows: {len(processed_data)}
            - Date range: {processed_data['Tanggal'].min().strftime('%Y-%m-%d')} to {processed_data['Tanggal'].max().strftime('%Y-%m-%d')}
            - Price range: {processed_data['Terakhir'].min():.2f} to {processed_data['Terakhir'].max():.2f}
            - Moving averages: MA{window_size1}, MA{window_size2}
            """)
        
        # Visualize preprocessed data
        st.subheader("📈 Historical Data with Moving Averages")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(processed_data['Tanggal'], processed_data['Terakhir'], 
                label='Closing Price', color='#4286f4', linewidth=2)
        ax.plot(processed_data['Tanggal'], processed_data[f'MA{window_size1}'], 
                label=f'MA{window_size1}', color='#ff8c00', linewidth=1.5)
        ax.plot(processed_data['Tanggal'], processed_data[f'MA{window_size2}'], 
                label=f'MA{window_size2}', color='#41ab5d', linewidth=1.5)
        
        ax.set_title(f'{stock_name} Historical Prices and Moving Averages', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Price (Rp)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        fig.tight_layout()
        st.pyplot(fig)

        # GPR Model Implementation
        if model_type == "GPR":
            # Prepare GPR features and split data
            with st.spinner("Preparing features for GPR model training..."):
                X, y, scaler_X, scaler_y, features = prepare_gpr_features(processed_data, window_size1, window_size2)
                X_train, X_val, X_test, y_train, y_val, y_test = split_training_data(X, y, len(processed_data), model_type="GPR")
                
                # Display data split info
                st.info(f"""
                **GPR Data Split:**
                - Training: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)
                - Validation: {X_val.shape[0]} samples ({X_val.shape[0]/X.shape[0]*100:.1f}%)
                - Testing: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)
                """)
                
            # Train GPR models
            st.subheader("🤖 GPR Model Training")
            
            # Check for saved model
            model_filename = f"model_gpr_{stock_name.lower()}.pkl"
            if os.path.exists(model_filename):
                load_saved = st.checkbox("Load previously saved GPR model?", value=True)
                
                if load_saved:
                    with open(model_filename, 'rb') as file:
                        model_data = pickle.load(file)
                        trained_models = model_data['models']
                        training_times = model_data['times']
                        best_model = model_data['best_model']
                        best_model_name = model_data['best_model_name']
                        val_results = model_data['val_results']
                        test_metrics = model_data['test_metrics']
                        train_metrics = model_data.get('train_metrics', None)  # Compatible with older saves
                    
                    st.success(f"Loaded saved GPR model from {model_filename}")
                    
                    # If no train metrics in older saved models
                    if train_metrics is None:
                        # Calculate train metrics
                        y_train_pred = best_model.predict(X_train)
                        train_metrics = calculate_metrics(y_train, y_train_pred, scaler_y, model_type="GPR")
                else:
                    # Train new models
                    with st.spinner("Training GPR models with different kernels..."):
                        trained_models, training_times = train_gpr_models(X_train, y_train)
                        best_model, best_model_name, val_results, test_metrics, train_metrics = evaluate_models(
                            trained_models, X_val, y_val, X_test, y_test, scaler_y, model_type="GPR")
                    
                    # Save model
                    model_data = {
                        'models': trained_models,
                        'times': training_times,
                        'best_model': best_model,
                        'best_model_name': best_model_name,
                        'val_results': val_results,
                        'test_metrics': test_metrics,
                        'train_metrics': train_metrics
                    }
                    
                    with open(model_filename, 'wb') as file:
                        pickle.dump(model_data, file)
                    
                    st.success(f"Saved GPR model to {model_filename}")
            else:
                # Train new models
                with st.spinner("Training GPR models with different kernels..."):
                    trained_models, training_times = train_gpr_models(X_train, y_train)
                    best_model, best_model_name, val_results, test_metrics, train_metrics = evaluate_models(
                        trained_models, X_val, y_val, X_test, y_test, scaler_y, model_type="GPR")
                
                # Save model
                model_data = {
                    'models': trained_models,
                    'times': training_times,
                    'best_model': best_model,
                    'best_model_name': best_model_name,
                    'val_results': val_results,
                    'test_metrics': test_metrics,
                    'train_metrics': train_metrics
                }
                
                with open(model_filename, 'wb') as file:
                    pickle.dump(model_data, file)
                
                st.success(f"Saved GPR model to {model_filename}")
            
            # Display model comparison
            st.subheader("📊 GPR Model Comparison")
            
            # Create metrics table
            metrics_df = pd.DataFrame()
            for name, metrics in val_results.items():
                metrics_df[name] = [
                    f"{metrics['MAE']:.2f}",
                    f"{metrics['RMSE']:.2f}",
                    f"{metrics['MAPE']:.2f}%",
                    f"{metrics['R2']:.4f}",
                    f"{training_times.get(name, 0):.2f} sec"
                ]
            
            metrics_df.index = ['MAE', 'RMSE', 'MAPE', 'R²', 'Training Time']
            st.table(metrics_df)
            
            # Display best model info
            st.success(f"""
            **Best GPR Model:** {best_model_name}
            - RMSE: {val_results[best_model_name]['RMSE']:.2f}
            - Optimized kernel: {best_model.kernel_}
            """)
            
            # Test set evaluation
            st.subheader("🔍 GPR Model Evaluation (Test Set)")
            
            # Display test metrics with colored boxes
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("MAE", f"{test_metrics['MAE']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("RMSE", f"{test_metrics['RMSE']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("MAPE", f"{test_metrics['MAPE']:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("R² Score", f"{test_metrics['R2']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Plot error analysis
            fig = plot_error_analysis(train_metrics, val_results[best_model_name], test_metrics)
            st.pyplot(fig)
            
            # Generate predictions
            with st.spinner("Generating GPR predictions..."):
                # Predict on full dataset
                y_pred_full = best_model.predict(X)
                
                # Convert back to original scale
                y_full_orig = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
                y_pred_full_orig = scaler_y.inverse_transform(y_pred_full.reshape(-1, 1)).flatten()
                
                # Get dates for visualization
                dates = processed_data['Tanggal'].values
                
                # Create visualization
                fig = plot_stock_predictions(
                    processed_data, dates, y_full_orig, y_pred_full_orig,
                    X_train.shape[0], X_val.shape[0], X_test.shape[0],
                    best_model_name, stock_name, model_type="GPR"
                )
                st.pyplot(fig)
            
            # Create comparison table
            st.subheader("📋 GPR Actual vs Predicted Prices (Test Set)")
            
            # Get test set data
            test_dates = dates[-X_test.shape[0]:]
            y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_test_pred_orig = scaler_y.inverse_transform(best_model.predict(X_test).reshape(-1, 1)).flatten()
            
            comparison_table = pd.DataFrame({
                'Date': test_dates,
                'Actual Price': y_test_orig,
                'Predicted Price': y_test_pred_orig,
                'Difference': np.abs(y_test_orig - y_test_pred_orig),
                'Error (%)': np.abs((y_test_orig - y_test_pred_orig) / y_test_orig) * 100
            })
            
            # Format table
            comparison_table['Date'] = pd.to_datetime(comparison_table['Date']).dt.strftime('%Y-%m-%d')
            comparison_table['Actual Price'] = comparison_table['Actual Price'].round(2)
            comparison_table['Predicted Price'] = comparison_table['Predicted Price'].round(2)
            comparison_table['Difference'] = comparison_table['Difference'].round(2)
            comparison_table['Error (%)'] = comparison_table['Error (%)'].round(2)
            
            st.dataframe(comparison_table)
            
            # Provide download link
            st.markdown(get_download_link(comparison_table, f"{stock_name}_gpr_comparison.csv"), unsafe_allow_html=True)
            
            # Future predictions
            st.subheader("🔮 GPR Future Price Predictions")
            
            future_days = st.slider("Number of days to predict:", 5, 60, 30, key="gpr_future_days")
            
            with st.spinner(f"Predicting prices for the next {future_days} days with GPR..."):
                # Generate future predictions
                future_predictions = predict_gpr_future(
                    best_model,
                    processed_data,
                    scaler_X,
                    scaler_y,
                    features,
                    future_days
                )
                
                # Format the dates for display
                future_predictions['Date'] = pd.to_datetime(future_predictions['Date']).dt.strftime('%Y-%m-%d')
                future_predictions['Predicted Price'] = future_predictions['Predicted Price'].round(2)
                
                # Display predictions
                st.dataframe(future_predictions)
                
                # Provide download link
                st.markdown(get_download_link(future_predictions, f"{stock_name}_gpr_future_predictions.csv"), unsafe_allow_html=True)
                
                # Plot future predictions
                fig = plot_future_predictions(
                    processed_data['Tanggal'].values,
                    processed_data['Terakhir'].values,
                    pd.to_datetime(future_predictions['Date']),
                    future_predictions['Predicted Price'].values,
                    stock_name,
                    model_type="GPR"
                )
                st.pyplot(fig)

        # LSTM Model Implementation
        else:  # LSTM with TensorFlow
            # Set sequence length for LSTM
            sequence_length = st.slider("LSTM Sequence Length (days to look back):", 10, 50, 30)
            
            # Prepare LSTM features and split data
            with st.spinner(f"Preparing features for LSTM model with sequence length {sequence_length}..."):
                X, y, scaler_X, scaler_y, seq_length, features = prepare_lstm_features(processed_data, sequence_length)
                X_train, X_val, X_test, y_train, y_val, y_test = split_training_data(X, y, len(X), model_type="LSTM")
                
                # Display data split info
                st.info(f"""
                **LSTM Data Split:**
                - Sequence length: {sequence_length} days
                - Training: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)
                - Validation: {X_val.shape[0]} samples ({X_val.shape[0]/X.shape[0]*100:.1f}%)
                - Testing: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)
                - Input shape: {X.shape[1]} time steps x {X.shape[2]} features
                """)
            
            # Train LSTM models
            st.subheader("🤖 LSTM Model Training")
            
            # Check for saved model
            model_filename = f"model_lstm_{stock_name.lower()}_{sequence_length}.pkl"
            if os.path.exists(model_filename):
                load_saved = st.checkbox("Load previously saved LSTM model?", value=True)
                
                if load_saved:
                    try:
                        with open(model_filename, 'rb') as file:
                            model_data = pickle.load(file)
                            trained_models = model_data.get('models', {})
                            training_times = model_data.get('times', {})
                            best_model = model_data.get('best_model', None)
                            best_model_name = model_data.get('best_model_name', '')
                            val_results = model_data.get('val_results', {})
                            test_metrics = model_data.get('test_metrics', {})
                            train_metrics = model_data.get('train_metrics', {})
                            histories = model_data.get('histories', {})
                        
                        st.success(f"Loaded saved LSTM model from {model_filename}")
                    except Exception as e:
                        st.error(f"Error loading saved model: {str(e)}")
                        st.warning("Training new LSTM models...")
                        
                        # Get input shape for LSTM
                        input_shape = (X_train.shape[1], X_train.shape[2])
                        
                        # Train new models
                        trained_models, training_times, histories = train_lstm_models(
                            X_train, y_train, X_val, y_val, input_shape)
                        best_model, best_model_name, val_results, test_metrics, train_metrics = evaluate_models(
                            trained_models, X_val, y_val, X_test, y_test, scaler_y, model_type="LSTM", histories=histories)
                        
                        # Save model
                        model_data = {
                            'models': trained_models,
                            'times': training_times,
                            'best_model': best_model,
                            'best_model_name': best_model_name,
                            'val_results': val_results,
                            'test_metrics': test_metrics,
                            'train_metrics': train_metrics,
                            'histories': histories
                        }
                        
                        with open(model_filename, 'wb') as file:
                            pickle.dump(model_data, file)
                        
                        st.success(f"Saved LSTM model to {model_filename}")
                else:
                    # Get input shape for LSTM
                    input_shape = (X_train.shape[1], X_train.shape[2])
                    
                    # Train new models
                    with st.spinner("Training LSTM models with different architectures..."):
                        trained_models, training_times, histories = train_lstm_models(
                            X_train, y_train, X_val, y_val, input_shape)
                        best_model, best_model_name, val_results, test_metrics, train_metrics = evaluate_models(
                            trained_models, X_val, y_val, X_test, y_test, scaler_y, model_type="LSTM", histories=histories)
                    
                    # Save model
                    model_data = {
                        'models': trained_models,
                        'times': training_times,
                        'best_model': best_model,
                        'best_model_name': best_model_name,
                        'val_results': val_results,
                        'test_metrics': test_metrics,
                        'train_metrics': train_metrics,
                        'histories': histories
                    }
                    
                    with open(model_filename, 'wb') as file:
                        pickle.dump(model_data, file)
                    
                    st.success(f"Saved LSTM model to {model_filename}")
            else:
                # Get input shape for LSTM
                input_shape = (X_train.shape[1], X_train.shape[2])
                
                # Train new models
                with st.spinner("Training LSTM models with different architectures..."):
                    trained_models, training_times, histories = train_lstm_models(
                        X_train, y_train, X_val, y_val, input_shape)
                    best_model, best_model_name, val_results, test_metrics, train_metrics = evaluate_models(
                        trained_models, X_val, y_val, X_test, y_test, scaler_y, model_type="LSTM", histories=histories)
                
                # Save model
                model_data = {
                    'models': trained_models,
                    'times': training_times,
                    'best_model': best_model,
                    'best_model_name': best_model_name,
                    'val_results': val_results,
                    'test_metrics': test_metrics,
                    'train_metrics': train_metrics,
                    'histories': histories
                }
                
                with open(model_filename, 'wb') as file:
                    pickle.dump(model_data, file)
                
                st.success(f"Saved LSTM model to {model_filename}")
            
            # Display model comparison
            st.subheader("📊 LSTM Model Comparison")
            
            # Create metrics table
            metrics_df = pd.DataFrame()
            for name, metrics in val_results.items():
                # Calculate number of parameters if model is a TensorFlow model
                n_params = trained_models[name].count_params() if hasattr(trained_models[name], 'count_params') else 0
                
                metrics_df[name] = [
                    f"{metrics['MAE']:.2f}",
                    f"{metrics['RMSE']:.2f}",
                    f"{metrics['MAPE']:.2f}%",
                    f"{metrics['R2']:.4f}",
                    f"{n_params:,}",
                    f"{training_times.get(name, 0):.2f} sec"
                ]
            
            metrics_df.index = ['MAE', 'RMSE', 'MAPE', 'R²', 'Parameters', 'Training Time']
            st.table(metrics_df)
            
            # Display best model info
            st.success(f"""
            **Best LSTM Model:** {best_model_name}
            - RMSE: {val_results[best_model_name]['RMSE']:.2f}
            - Parameters: {best_model.count_params():,}
            """)
            
            # Test set evaluation
            st.subheader("🔍 LSTM Model Evaluation (Test Set)")
            
            # Display test metrics with colored boxes
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("MAE", f"{test_metrics['MAE']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("RMSE", f"{test_metrics['RMSE']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("MAPE", f"{test_metrics['MAPE']:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("R² Score", f"{test_metrics['R2']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Plot error analysis
            fig = plot_error_analysis(train_metrics, val_results[best_model_name], test_metrics)
            st.pyplot(fig)
            
            # Plot training history
            st.subheader("📉 LSTM Training History")
            
            if histories:
                fig = plot_lstm_training_history(histories, best_model_name)
                st.pyplot(fig)
            
            # Generate predictions
            with st.spinner("Generating LSTM predictions..."):
                # Dates for LSTM are shifted due to sequence length
                dates = processed_data['Tanggal'].values[sequence_length:]
                
                # Predict on full dataset using TensorFlow model
                y_pred_full = best_model.predict(X, verbose=0)
                
                # Convert back to original scale
                y_full_orig = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
                y_pred_full_orig = scaler_y.inverse_transform(y_pred_full).flatten()
                
                # Create visualization
                fig = plot_stock_predictions(
                    processed_data, dates, y_full_orig, y_pred_full_orig,
                    X_train.shape[0], X_val.shape[0], X_test.shape[0],
                    best_model_name, stock_name, model_type="LSTM"
                )
                st.pyplot(fig)
            
            # Create comparison table
            st.subheader("📋 LSTM Actual vs Predicted Prices (Test Set)")
            
            # Get test set data
            test_dates = dates[-X_test.shape[0]:]
            
            # Make predictions on test data
            y_test_pred = best_model.predict(X_test, verbose=0)
            
            # Convert to original scale
            y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_test_pred_orig = scaler_y.inverse_transform(y_test_pred).flatten()
            
            # Create comparison table
            comparison_table = pd.DataFrame({
                'Date': test_dates,
                'Actual Price': y_test_orig,
                'Predicted Price': y_test_pred_orig,
                'Difference': np.abs(y_test_orig - y_test_pred_orig),
                'Error (%)': np.abs((y_test_orig - y_test_pred_orig) / y_test_orig) * 100
            })
            
            # Format table
            comparison_table['Date'] = pd.to_datetime(comparison_table['Date']).dt.strftime('%Y-%m-%d')
            comparison_table['Actual Price'] = comparison_table['Actual Price'].round(2)
            comparison_table['Predicted Price'] = comparison_table['Predicted Price'].round(2)
            comparison_table['Difference'] = comparison_table['Difference'].round(2)
            comparison_table['Error (%)'] = comparison_table['Error (%)'].round(2)
            
            st.dataframe(comparison_table)
            
            # Provide download link
            st.markdown(get_download_link(comparison_table, f"{stock_name}_lstm_comparison.csv"), unsafe_allow_html=True)
            
            # Future predictions
            st.subheader("🔮 LSTM Future Price Predictions")
            
            future_days = st.slider("Number of days to predict:", 5, 60, 30, key="lstm_future_days")
            
            with st.spinner(f"Predicting prices for the next {future_days} days with LSTM..."):
                # Generate future predictions using LSTM model
                future_predictions = predict_lstm_future(
                    best_model,
                    sequence_length,
                    processed_data,
                    scaler_X,
                    scaler_y,
                    features,
                    future_days
                )
                
                # Format the dates for display
                future_predictions['Date'] = pd.to_datetime(future_predictions['Date']).dt.strftime('%Y-%m-%d')
                future_predictions['Predicted Price'] = future_predictions['Predicted Price'].round(2)
                
                # Display predictions
                st.dataframe(future_predictions)
                
                # Provide download link
                st.markdown(get_download_link(future_predictions, f"{stock_name}_lstm_future_predictions.csv"), unsafe_allow_html=True)
                
                # Plot future predictions
                fig = plot_future_predictions(
                    processed_data['Tanggal'].values,
                    processed_data['Terakhir'].values,
                    pd.to_datetime(future_predictions['Date']),
                    future_predictions['Predicted Price'].values,
                    stock_name,
                    model_type="LSTM"
                )
                st.pyplot(fig)

        # Success message
        st.success(f"Analysis complete for {stock_name} stock using {model_type} model!")
        
    except Exception as e:
        st.error(f"Error in prediction process: {str(e)}")
        st.error("Please check your input file and try again.")
        import traceback
        st.error(traceback.format_exc())

def display_oos_evaluation():
    """
    Display Out of Sample Evaluation page
    """
    st.title("🔍 Out-of-Sample Evaluation")
    
    st.markdown("""
    <div style="background-color: #f5f7f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3>📊 Compare Predictions with Actual Values</h3>
        <p>Evaluate how well the model predictions match actual stock prices outside of the training dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploaders for prediction data and actual data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Prediction Data")
        predictions_file = st.file_uploader("Upload model predictions (CSV):", type=["csv"], key="predictions_file")
        
    with col2:
        st.subheader("Upload Actual Data")
        actual_file = st.file_uploader("Upload actual stock data (CSV):", type=["csv"], key="actual_file")
    
    if predictions_file is not None and actual_file is not None:
        try:
            # Load prediction data
            predictions_df = pd.read_csv(predictions_file)
            
            # Load actual data
            actual_df = pd.read_csv(actual_file, thousands='.')
            
            # Check for required columns in prediction data
            if 'Date' not in predictions_df.columns or 'Predicted Price' not in predictions_df.columns:
                st.error("Prediction file must have 'Date' and 'Predicted Price' columns")
                return
            
            # Check for required columns in actual data
            if 'Tanggal' not in actual_df.columns or 'Terakhir' not in actual_df.columns:
                st.error("Actual data file must have 'Tanggal' and 'Terakhir' columns")
                return
            
            # Convert date columns to datetime
            predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
            
            try:
                # Try DD/MM/YYYY format first
                actual_df['Tanggal'] = pd.to_datetime(actual_df['Tanggal'], dayfirst=True)
            except ValueError:
                try:
                    # Try MM/DD/YYYY format
                    actual_df['Tanggal'] = pd.to_datetime(actual_df['Tanggal'], dayfirst=False)
                except ValueError:
                    # Let pandas guess the format
                    actual_df['Tanggal'] = pd.to_datetime(actual_df['Tanggal'], errors='coerce')
            
            # Sort both dataframes by date
            predictions_df = predictions_df.sort_values('Date')
            actual_df = actual_df.sort_values('Tanggal')
            
            # Convert to numeric if needed
            actual_df['Terakhir'] = pd.to_numeric(actual_df['Terakhir'], errors='coerce')
            
            # Merge the dataframes on date
            merged_df = pd.merge_asof(
                predictions_df,
                actual_df[['Tanggal', 'Terakhir']].rename(columns={'Tanggal': 'Date', 'Terakhir': 'Actual Price'}),
                on='Date',
                direction='nearest'
            )
            
            # Drop rows with missing actual prices
            merged_df = merged_df.dropna(subset=['Actual Price'])
            
            # Calculate error metrics
            merged_df['Absolute Error'] = abs(merged_df['Predicted Price'] - merged_df['Actual Price'])
            merged_df['Percentage Error'] = (merged_df['Absolute Error'] / merged_df['Actual Price']) * 100
            
            # Display merged data
            st.subheader("📋 Prediction vs Actual Data")
            st.dataframe(merged_df)
            
            # Provide download link for the merged data
            st.markdown(get_download_link(merged_df, "prediction_vs_actual.csv"), unsafe_allow_html=True)
            
            # Create evaluation periods
            st.subheader("📊 Evaluation Across Different Time Periods")
            
            # Get max number of days available
            max_days = len(merged_df)
            
            # Define evaluation periods (3 days, 7 days, 30 days)
            evaluation_periods = {
                "3 Days": min(3, max_days),
                "7 Days": min(7, max_days),
                "30 Days": min(30, max_days)
            }
            
            # Calculate metrics for each period
            metrics_df = pd.DataFrame(index=["MAE", "RMSE", "MAPE", "R²"])
            
            for period_name, days in evaluation_periods.items():
                period_data = merged_df.head(days)
                
                # Calculate metrics
                mae = mean_absolute_error(period_data['Actual Price'], period_data['Predicted Price'])
                mse = mean_squared_error(period_data['Actual Price'], period_data['Predicted Price'])
                rmse = np.sqrt(mse)
                
                # Calculate MAPE with handling for zero values
                epsilon = 1e-10  # Prevent division by zero
                mape = np.mean(np.abs((period_data['Actual Price'] - period_data['Predicted Price']) / 
                                      (period_data['Actual Price'] + epsilon))) * 100
                
                # Calculate R² score
                r2 = r2_score(period_data['Actual Price'], period_data['Predicted Price'])
                
                # Add to metrics DataFrame
                metrics_df[period_name] = [
                    f"{mae:.2f}",
                    f"{rmse:.2f}",
                    f"{mape:.2f}%",
                    f"{r2:.4f}"
                ]
            
            # Display metrics table
            st.table(metrics_df)
            
            # Visualize predictions vs actual values
            st.subheader("📈 Visualization: Predicted vs Actual Prices")
            
            # Create tabs for different periods
            tabs = st.tabs(list(evaluation_periods.keys()))
            
            for i, (period_name, days) in enumerate(evaluation_periods.items()):
                with tabs[i]:
                    period_data = merged_df.head(days)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot actual vs predicted values
                    ax.plot(period_data['Date'], period_data['Actual Price'], 
                           label='Actual', color='#4286f4', linewidth=2, marker='o')
                    ax.plot(period_data['Date'], period_data['Predicted Price'], 
                           label='Predicted', color='#ff8c00', linewidth=2, marker='x', linestyle='--')
                    
                    # Add labels and formatting
                    ax.set_title(f'Predicted vs Actual Prices ({period_name})', fontsize=16)
                    ax.set_xlabel('Date', fontsize=14)
                    ax.set_ylabel('Price (Rp)', fontsize=14)
                    ax.legend(loc='best')
                    ax.grid(True, alpha=0.3)
                    
                    # Format dates
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.xticks(rotation=45)
                    
                    fig.tight_layout()
                    st.pyplot(fig)
                    
                    # Display percentage error analysis
                    st.subheader(f"Error Analysis ({period_name})")
                    
                    # Error distribution
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.bar(period_data['Date'], period_data['Percentage Error'], color='#e63946', alpha=0.7)
                    ax.set_title(f'Percentage Error by Date ({period_name})', fontsize=16)
                    ax.set_xlabel('Date', fontsize=14)
                    ax.set_ylabel('Percentage Error (%)', fontsize=14)
                    ax.grid(axis='y', alpha=0.3)
                    
                    # Format dates
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.xticks(rotation=45)
                    
                    fig.tight_layout()
                    st.pyplot(fig)
                    
                    # Error metrics for the period
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Average % Error", f"{period_data['Percentage Error'].mean():.2f}%")
                    
                    with col2:
                        st.metric("Max % Error", f"{period_data['Percentage Error'].max():.2f}%")
                    
                    with col3:
                        st.metric("Min % Error", f"{period_data['Percentage Error'].min():.2f}%")
                    
                    with col4:
                        st.metric("Std Dev of % Error", f"{period_data['Percentage Error'].std():.2f}%")
            
            # Add error trend analysis
            st.subheader("📉 Error Trend Analysis")
            
            # Plot error trend
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Rolling average of percentage error
            window_size = min(5, max_days)
            merged_df['Rolling Avg Error'] = merged_df['Percentage Error'].rolling(window=window_size).mean()
            
            ax.plot(merged_df['Date'], merged_df['Percentage Error'], 
                   label='% Error', color='#e63946', alpha=0.5, marker='o')
            ax.plot(merged_df['Date'], merged_df['Rolling Avg Error'], 
                   label=f'{window_size}-day Rolling Avg', color='#1d3557', linewidth=2)
            
            # Add labels and formatting
            ax.set_title('Error Trend Over Time', fontsize=16)
            ax.set_xlabel('Date', fontsize=14)
            ax.set_ylabel('Percentage Error (%)', fontsize=14)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            fig.tight_layout()
            st.pyplot(fig)
            
            # Summary conclusion
            st.subheader("📋 Evaluation Summary")
            
            # Generate summary based on overall MAPE
            overall_mape = merged_df['Percentage Error'].mean()
            
            if overall_mape < 5:
                evaluation = "Excellent"
                description = "The model demonstrates excellent predictive accuracy with very low error rates."
            elif overall_mape < 10:
                evaluation = "Good"
                description = "The model shows good predictive performance with acceptable error rates."
            elif overall_mape < 15:
                evaluation = "Moderate"
                description = "The model shows moderate predictive accuracy, with room for improvement."
            else:
                evaluation = "Poor"
                description = "The model demonstrates poor predictive accuracy and needs significant improvement."
            
            st.info(f"""
            **Out of Sample Evaluation: {evaluation}**
            
            {description}
            
            - Overall Mean Absolute Percentage Error (MAPE): {overall_mape:.2f}%
            - Best performing period: {metrics_df.loc['MAPE'].astype(str).str.rstrip('%').astype(float).idxmin()} (MAPE: {metrics_df.loc['MAPE'][metrics_df.loc['MAPE'].astype(str).str.rstrip('%').astype(float).idxmin()]})
            - Worst performing period: {metrics_df.loc['MAPE'].astype(str).str.rstrip('%').astype(float).idxmax()} (MAPE: {metrics_df.loc['MAPE'][metrics_df.loc['MAPE'].astype(str).str.rstrip('%').astype(float).idxmax()]})
            
            **Recommendations:**
            - Consider model retraining if error trend is increasing over time
            - Monitor predictions regularly against actual values
            - Focus on improving {metrics_df.loc['MAPE'].astype(str).str.rstrip('%').astype(float).idxmax()} forecasts
            """)
            
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    else:
        st.info("Please upload both prediction and actual data files to begin the evaluation.")

def display_historical_data():
    """
    Display historical stock data with flexible stock input
    """
    st.title("📜 Historical Stock Data")
    
    st.markdown("""
    <div style="background-color: #f5f7f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3>📊 Browse Historical Stock Data</h3>
        <p>View and analyze historical stock price data for any stock.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Option to enter custom stock name or choose from popular ones
    use_custom = st.checkbox("Enter custom stock name", value=False)
    
    if use_custom:
        # Custom stock input
        selected_stock = st.text_input("Enter stock code (e.g., BMRI, BBRI):", "").upper()
        if not selected_stock:
            st.warning("Please enter a stock code to continue.")
            return
    else:
        # Suggestion of popular stocks (but not limited to these)
        sample_stocks = ["BMRI", "BBRI", "BBNI", "BBCA"]
        selected_stock = st.selectbox("Select a popular stock or choose 'Enter custom stock name' for others:", sample_stocks)
    
    # Load or upload data
    if not selected_stock:
        st.warning("Please select or enter a stock code to continue.")
        return
    
    placeholder = st.empty()
    placeholder.info(f"Loading historical data for {selected_stock}...")
    
    try:
        # Try to load from file first
        filename = f"Data_Historis_{selected_stock}.csv"
        if os.path.exists(filename):
            # Gunakan parameter thousands='.' untuk format angka Indonesia
            df = pd.read_csv(filename, thousands='.')
            
            # Convert date - try multiple formats with error handling
            try:
                # Try multiple date formats with explicit parsing
                try:
                    # Try DD/MM/YYYY format first (Indonesian style)
                    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y')
                except:
                    try:
                        # Try MM/DD/YYYY format (US style)
                        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%m/%d/%Y')
                    except:
                        try:
                            # Try YYYY-MM-DD format (ISO)
                            df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%Y-%m-%d')
                        except:
                            # Last resort - let pandas guess
                            df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
            except Exception as date_err:
                st.warning(f"Date conversion issue: {date_err}. Some dates may not display correctly.")
                
            # Ensure numeric values
            if 'Terakhir' in df.columns:
                df['Terakhir'] = pd.to_numeric(df['Terakhir'], errors='coerce')
            
            placeholder.success(f"Loaded data for {selected_stock}")
            
            # Display data
            st.subheader(f"{selected_stock} Historical Data")
            st.dataframe(df)
            
            # Plot price history (only if dates were successfully converted)
            if not pd.isna(df['Tanggal']).all():
                st.subheader(f"{selected_stock} Price History")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df['Tanggal'], df['Terakhir'], marker='o', linestyle='-', color='#4286f4')
                ax.set_title(f'{selected_stock} Historical Stock Prices', fontsize=16)
                ax.set_xlabel('Date', fontsize=14)
                ax.set_ylabel('Price (Rp)', fontsize=14)
                ax.grid(True, alpha=0.3)
                
                # Format axes
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                fig.tight_layout()
                
                st.pyplot(fig)
                
                # Statistics
                st.subheader(f"{selected_stock} Statistics")
                
                # Calculate statistics
                stats = df['Terakhir'].describe()
                
                # Display in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average Price", f"{stats['mean']:.2f}")
                
                with col2:
                    st.metric("Min Price", f"{stats['min']:.2f}")
                
                with col3:
                    st.metric("Max Price", f"{stats['max']:.2f}")
                
                with col4:
                    # Calculate price change
                    if len(df) >= 2:
                        first_price = df['Terakhir'].iloc[0]
                        last_price = df['Terakhir'].iloc[-1]
                        price_change = ((last_price - first_price) / first_price) * 100
                        st.metric("Price Change", f"{price_change:.2f}%")
            else:
                st.error("Could not plot the data due to date conversion issues.")
            
            # Provide download link
            st.markdown(get_download_link(df, f"{selected_stock}_historical_data.csv"), unsafe_allow_html=True)
            
        else:
            placeholder.warning(f"No historical data found for {selected_stock}. Please upload data.")
            
            # Upload option
            uploaded_file = st.file_uploader(f"Upload {selected_stock} historical data:", type=["csv", "xlsx"])
            
            if uploaded_file is not None:
                if uploaded_file.name.lower().endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    # Gunakan parameter thousands='.' untuk format angka Indonesia
                    df = pd.read_csv(uploaded_file, thousands='.')
                
                st.success("File uploaded successfully!")
                st.dataframe(df.head())
                
                # Verify data format before saving
                if 'Tanggal' not in df.columns or 'Terakhir' not in df.columns:
                    st.error("Uploaded file must contain 'Tanggal' (Date) and 'Terakhir' (Closing Price) columns.")
                else:
                    # Save option with auto-confirmation
                    if st.button(f"Save as {selected_stock} historical data"):
                        df.to_csv(filename, index=False)
                        st.success(f"Data saved successfully as {filename}. You can now load this data anytime by entering {selected_stock}.")
                        st.rerun()
                
    except Exception as e:
        placeholder.error(f"Error loading data: {str(e)}")
        st.error("Please ensure your data has the correct format: 'Tanggal' column with dates and 'Terakhir' column with prices.")

def display_about():
    """
    Display About page
    """
    st.title("ℹ️ About")
    
    # Profile section
    st.markdown("""
    <div style="background-color: #f5f7f9; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
        <h2>Gaussian Process Regression Prediction - GPRO</h2>
        <p style="font-style: italic;">Developed by Raja Valentino Kristananda</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Application overview
    st.markdown("""
    ### 📊 Application Overview
    
    This application uses advanced machine learning techniques to predict stock prices based on historical data. 
    You can choose between two powerful models:
    
    - **Gaussian Process Regression (GPR)**: A flexible non-parametric approach that excels at uncertainty estimation
    - **Long Short-Term Memory (LSTM)**: A deep learning approach that can capture complex temporal patterns in time series data
    
    #### Key Features:
    
    - **Multiple Model Options**: Choose between GPR and LSTM based on your prediction needs
    - **GPR with Multiple Kernels**: Compares RBF, Matern, RationalQuadratic, and Linear kernels to find the best fit
    - **LSTM with Various Architectures**: Tests different layer configurations and hyperparameters
    - **Comprehensive Evaluation**: Uses MAE, RMSE, MAPE, and R² metrics to evaluate model performance
    - **Visualizations**: Visual representation of predictions, model performance, and training progress
    - **Future Predictions**: Generate predictions for up to 60 days into the future
    - **Historical Data Analysis**: Browse and analyze historical stock data
    - **Out-of-Sample Evaluation**: Compare predictions with actual values outside the training dataset
    
    ### 🧠 Machine Learning Approaches
    
    #### Gaussian Process Regression
    - Provides uncertainty estimates for predictions
    - Adapts well to different data patterns through kernel selection
    - Works effectively with smaller datasets
    - Requires minimal hyperparameter tuning
    
    #### Long Short-Term Memory Networks
    - Captures long-term dependencies in time series data
    - Can learn complex non-linear relationships
    - Excels at sequential data prediction
    - Particularly effective with larger datasets
    
    ### 📈 Input Data Format
    
    The application accepts stock data in CSV or Excel format with the following required columns:
    
    - **Tanggal**: Date column (date format)
    - **Terakhir**: Closing price (numeric)
    
    Additional columns like opening price, high, low, and volume may be present but are not used for prediction.
    """)
    
    # Contact/Questions section
    st.markdown("""
    ### 📞 Questions or Feedback?
    
    For questions, feedback, or suggestions, please contact:
    
    - **Email**: rajavalentinokristananda@gmail.com
    - **GitHub**: github.com/RajaValentinoKristananda
    
    ### 📝 License
    
    This project is open source and available under the MIT License.
    """)

def display_home():
    """
    Display homepage
    """
    st.title("📊 Gaussian Process Regression Prediction - GPRO")
    
    st.markdown("""
    <div style="background-color: #f5f7f9; padding: 20px; border-radius: 10px; margin-bottom: 30px;">
        <h2>Welcome to GPRO!</h2>
        <p>A machine learning application to predict stock prices using Gaussian Process Regression (GPR) or Long Short-Term Memory (LSTM) networks.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main dashboard cards with gradient backgrounds and improved styling
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="dashboard-card big-dashboard-card predict-card">
            <div class="big-card-icon">🔮</div>
            <h2 class="big-card-title">Predict Stock Prices</h2>
            <p class="big-card-desc">Upload data and predict future stock prices</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Predict Stock Prices", key="go_predict"):
            st.session_state.page = "Predict Stock Prices"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="dashboard-card big-dashboard-card oos-card">
            <div class="big-card-icon">🔍</div>
            <h2 class="big-card-title">Out-of-Sample Evaluation</h2>
            <p class="big-card-desc">Evaluate model performance with real-world data</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Out-of-Sample Evaluation", key="go_oos"):
            st.session_state.page = "Out-of-Sample Evaluation"
            st.rerun()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class="dashboard-card big-dashboard-card historical-card">
            <div class="big-card-icon">📊</div>
            <h2 class="big-card-title">Historical Data</h2>
            <p class="big-card-desc">Browse historical stock price data</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Historical Data", key="go_historical"):
            st.session_state.page = "Historical Data"
            st.rerun()
    
    with col4:
        st.markdown("""
        <div class="dashboard-card big-dashboard-card about-card">
            <div class="big-card-icon">ℹ️</div>
            <h2 class="big-card-title">About</h2>
            <p class="big-card-desc">Learn about the application and methodology</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to About", key="go_about"):
            st.session_state.page = "About"
            st.rerun()
    
    # Feature highlights
    st.markdown("### ✨ Key Features")
    
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        - **Multiple Model Options**: Choose between GPR and LSTM models
        - **Interactive Visualizations**: Visual representation of predictions
        - **Comprehensive Metrics**: Detailed model performance evaluation
        - **Out-of-Sample Testing**: Evaluate on real market data
        """)
    
    with features_col2:
        st.markdown("""
        - **Future Predictions**: Generate price forecasts for up to 60 days
        - **Historical Analysis**: Explore past price patterns and trends
        - **Export Results**: Download predictions and analysis as CSV
        - **Model Comparisons**: Compare performance across different models
        """)
    
    # Getting started section
    st.markdown("### 🚀 Getting Started")
    
    st.markdown("""
    To get started with GPRO:
    
    1. **Navigate to 'Predict Stock Prices'** section 
    2. **Upload** your historical stock data CSV/Excel file
    3. **Enter** the stock name
    4. **Choose** between GPR and LSTM models
    5. **Run the prediction** to see results and future forecasts
    6. **Evaluate performance** using the Out-of-Sample Evaluation feature
    """)
    
    # Add a footer
    st.markdown("""
    <div class="footer">
        <p>Developed by Raja Valentino Kristananda | Powered by Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# Main function
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    # Create styled sidebar and get selected page
    current_page = styled_sidebar(st.session_state.page)
    
    # Update session state if page changed from sidebar
    if current_page != st.session_state.page:
        st.session_state.page = current_page
        st.rerun()
    
    # Handle page selection
    if st.session_state.page == "Home":
        display_home()
    
    elif st.session_state.page == "Predict Stock Prices":
        st.title("📈 Predict Stock Prices")
        
        # File uploader and stock name input
        uploaded_file = st.file_uploader("Upload stock data (CSV or Excel):", type=["csv", "xlsx"])
        stock_name = st.text_input("Enter stock name (e.g., BMRI):", "BMRI")
        
        # Model selection
        st.subheader("Select Prediction Model")
        
        model_option = st.radio(
            "Choose a prediction model:",
            ["Gaussian Process Regression (GPR)", "Long Short-Term Memory (LSTM)"],
            help="GPR works better with smaller datasets. LSTM is better for capturing complex patterns in larger datasets."
        )
        
        # Determine model type from selection
        model_type = "GPR" if "Gaussian" in model_option else "LSTM"
        
        # Model description based on selection
        if model_type == "GPR":
            st.info("""
            **Gaussian Process Regression (GPR)** is a non-parametric model that provides uncertainty estimates and works well even with smaller datasets. 
            The app will compare different kernels (RBF, Matern, RationalQuadratic, Linear) to find the best fit for your data.
            """)
        else:
            st.info("""
            **Long Short-Term Memory (LSTM)** is a deep learning approach that excels at capturing complex temporal patterns in time series data.
            The app will compare different LSTM architectures to find the best configuration for your stock data.
            """)
        
        # Run prediction if all inputs are provided
        if uploaded_file is not None and stock_name:
            if st.button("Start Prediction", key="start_prediction"):
                with st.spinner(f"Running {model_type} prediction..."):
                    predict_stock_prices(uploaded_file, stock_name, model_type)
        else:
            st.info("Please upload a file and enter a stock name to begin.")
    
    elif st.session_state.page == "Out-of-Sample Evaluation":
        display_oos_evaluation()
    
    elif st.session_state.page == "Historical Data":
        display_historical_data()
    
    elif st.session_state.page == "About":
        display_about()

# Run the app
if __name__ == "__main__":
    main()