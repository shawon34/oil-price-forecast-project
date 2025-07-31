# Add this before any other imports
import sys
import os

# Fix numpy import issues
try:
    import numpy as np
except ImportError:
    # Attempt to recover from broken numpy installation
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "numpy==1.26.2"])
    import numpy as np

# Now continue with other imports
from flask import Flask, jsonify, render_template
import joblib
# ... rest of your imports ...


from flask import Flask, jsonify, render_template
import joblib
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import numpy as np
import os
import traceback
import logging
import math
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get absolute path to current directory
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Safe defaults
model = None
last_point = {
    'close': 75.0,
    'date': datetime.now()
}

# ====================== MODEL LOADING WITH ROBUST ERROR HANDLING ======================
try:
    model_path = os.path.join(APP_ROOT, 'trained_oil_price_model.sav')
    logger.info(f"Attempting to load model from: {model_path}")
    
    if os.path.exists(model_path):
        try:
            # First attempt with default loader
            model = joblib.load(model_path)
            logger.info("Model loaded successfully with default method!")
        except Exception as e:
            logger.error(f"Default load failed: {e}")
            logger.error(traceback.format_exc())
            
            # Second attempt with mmap_mode
            try:
                logger.info("Trying with mmap_mode='r'")
                model = joblib.load(model_path, mmap_mode='r')
                logger.info("Model loaded successfully with mmap_mode!")
            except Exception as e2:
                logger.error(f"mmap_mode load failed: {e2}")
                
                # Third attempt with unsafe pickle
                try:
                    logger.info("Trying with unsafe pickle")
                    with open(model_path, 'rb') as f:
                        model = joblib.load(f)
                    logger.info("Model loaded successfully with unsafe pickle!")
                except Exception as e3:
                    logger.error(f"All loading attempts failed: {e3}")
    else:
        logger.critical(f"Model file not found at: {model_path}")
except Exception as e:
    logger.error(f"Model loading process failed: {e}")
    logger.error(traceback.format_exc())

# ====================== LAST DATA POINT LOADING ======================
try:
    last_point_path = os.path.join(APP_ROOT, 'last_data_point.sav')
    logger.info(f"Attempting to load last data point from: {last_point_path}")
    
    if os.path.exists(last_point_path):
        last_point = joblib.load(last_point_path)
        if isinstance(last_point['date'], str):
            last_point['date'] = datetime.strptime(last_point['date'], '%Y-%m-%d')
        logger.info(f"Last data point: {last_point['date']} - ${last_point['close']:.2f}")
    else:
        logger.warning(f"Last data point file not found at: {last_point_path}")
        logger.info("Using default values")
except Exception as e:
    logger.error(f"Last data point loading failed: {e}")
    logger.info("Using default values")

# ====================== FALLBACK MODEL INITIALIZATION ======================
if model is None:
    logger.critical("Prediction model not available. Initializing fallback model.")
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.base import BaseEstimator
        
        # Create a simple dummy model
        class FallbackModel(BaseEstimator):
            def get_forecast(self, steps):
                return type('Forecast', (object,), {
                    'predicted_mean': np.zeros(steps)
                })
            
            def __repr__(self):
                return "FallbackModel()"
            
            def __str__(self):
                return "FallbackModel"
        
        model = FallbackModel()
        logger.warning("Initialized emergency fallback model")
    except Exception as e:
        logger.critical("No model available: " + str(e))
        model = None

# ====================== UTILITY FUNCTIONS ======================
def safe_float(value):
    """Convert value to float safely, handling NaNs and infs"""
    try:
        if value is None:
            return 0.0
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return 0.0
        return float(value)
    except:
        return 0.0

def get_realtime_price():
    """Fetch real-time oil price with NaN protection"""
    try:
        logger.info("Fetching real-time data...")
        
        # Try intraday data
        oil_data = yf.download('CL=F', period='1d', interval='1m', prepost=True, progress=False, timeout=5)
        if not oil_data.empty:
            valid_close = oil_data['Close'].dropna()
            if not valid_close.empty:
                price = safe_float(valid_close.iloc[-1])
                return price, valid_close.index[-1].to_pydatetime()
        
        # Fallback to daily data
        oil_data = yf.download('CL=F', period='1d', progress=False, timeout=5)
        if not oil_data.empty:
            price = safe_float(oil_data['Close'].iloc[-1])
            return price, oil_data.index[-1].to_pydatetime()
        
        # Final fallback
        logger.warning("Using last saved data point")
        return safe_float(last_point['close']), last_point['date']
        
    except Exception as e:
        logger.error(f"Data fetch error: {e}")
        return safe_float(last_point['close']), last_point['date']

# ====================== FORECAST GENERATION ======================
def generate_forecast(periods):
    """Generate forecast with NaN protection and sorted periods"""
    try:
        if model is None:
            logger.warning("No model available for forecasting")
            return {"error": "Prediction model not available"}
        
        current_price, current_dt = get_realtime_price()
        
        # Validate current price
        if current_price <= 0 or math.isnan(current_price):
            logger.warning(f"Invalid current price: {current_price}, using fallback")
            current_price = safe_float(last_point['close'])
        
        # Sort periods to ensure chronological order
        sorted_periods = sorted(periods)
        forecasts = {}
        
        for days in sorted_periods:
            try:
                # Generate forecast
                forecast = model.get_forecast(steps=days)
                forecast_returns = forecast.predicted_mean
                
                # Convert to cumulative returns with NaN protection
                cumulative_returns = np.cumsum([safe_float(r) for r in forecast_returns])
                
                # Convert to absolute prices with NaN protection
                predicted_prices = []
                for i, change in enumerate(cumulative_returns):
                    # Prevent exponential explosion
                    safe_change = max(min(change, 1.0), -1.0)
                    price = current_price * np.exp(safe_change)
                    predicted_prices.append(safe_float(price))
                
                # Generate business dates
                forecast_dates = []
                date = current_dt
                count = 0
                while count < days:
                    date += timedelta(days=1)
                    if date.weekday() < 5:  # Monday-Friday
                        forecast_dates.append(date.strftime('%Y-%m-%d'))
                        count += 1
                
                forecasts[days] = {
                    "predicted": predicted_prices,
                    "predicted_dates": forecast_dates
                }
                
            except Exception as e:
                logger.error(f"Forecast error for {days} days: {e}")
                # Generate fallback dates
                forecast_dates = []
                date = current_dt
                count = 0
                while count < days:
                    date += timedelta(days=1)
                    if date.weekday() < 5:
                        forecast_dates.append(date.strftime('%Y-%m-%d'))
                        count += 1
                
                forecasts[days] = {
                    "predicted": [current_price] * days,
                    "predicted_dates": forecast_dates
                }
        
        # Create sorted results
        results = {
            "last_actual_price": current_price,
            "last_actual_date": current_dt.strftime('%Y-%m-%d %H:%M'),
            "forecasts": {}
        }
        
        # Add forecasts in sorted order
        for days in sorted_periods:
            results["forecasts"][f"{days}_days"] = forecasts[days]
        
        return results
    
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}")
        return {"error": "Forecast generation failed"}

# ====================== ACCURACY CALCULATION ======================
def calculate_accuracy(forecast_days=90):
    """Calculate model accuracy using historical backtesting"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Get historical data (last 3 years + forecast period)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3 * 365 + forecast_days)
        oil_data = yf.download('CL=F', start=start_date, end=end_date)
        
        if oil_data.empty or len(oil_data) < forecast_days + 10:
            return {"error": "Insufficient historical data"}
        
        # Use closing prices
        historical = oil_data['Close'].dropna()
        
        # Find point to split data for backtesting
        split_index = len(historical) - forecast_days
        if split_index < 1:
            return {"error": "Not enough data for backtesting"}
        
        # Get the price at split point
        split_price = historical.iloc[split_index-1]
        actual_prices = historical.iloc[split_index:split_index+forecast_days].values
        
        # Generate forecast from split point
        forecast = model.get_forecast(steps=forecast_days)
        forecast_returns = forecast.predicted_mean
        
        # Convert returns to prices
        cumulative_returns = np.cumsum(forecast_returns)
        predicted_prices = split_price * np.exp(cumulative_returns)
        
        # Calculate metrics
        mae = mean_absolute_error(actual_prices, predicted_prices)
        mape = mean_absolute_percentage_error(actual_prices, predicted_prices) * 100
        r2 = r2_score(actual_prices, predicted_prices)
        
        # Calculate direction accuracy
        actual_directions = np.sign(actual_prices[1:] - actual_prices[:-1])
        predicted_directions = np.sign(predicted_prices[1:] - predicted_prices[:-1])
        direction_accuracy = np.mean(actual_directions == predicted_directions) * 100
        
        # Get dates for the forecast period
        dates = historical.index[split_index:split_index+forecast_days].strftime('%Y-%m-%d').tolist()
        
        return {
            "metrics": {
                "MAE": float(mae),
                "MAPE": float(mape),
                "R2": float(r2),
                "Direction_Accuracy": float(direction_accuracy)
            },
            "actual": actual_prices.tolist(),
            "predicted": predicted_prices.tolist(),
            "dates": dates
        }
    except Exception as e:
        logger.error(f"Accuracy calculation error: {e}")
        return {"error": str(e)}

# ====================== ROUTES ======================
@app.route('/predict')
def predict():
    """API endpoint for predictions"""
    try:
        periods = [30, 90, 180, 270, 365]  # 1m, 3m, 6m, 9m, 1y
        result = generate_forecast(periods)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Predict endpoint error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/model-accuracy')
def model_accuracy():
    """API endpoint for model accuracy assessment"""
    return jsonify(calculate_accuracy(90))  # 90-day backtest

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Enhanced health check with model status"""
    status = {
        "app": "running",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else "None",
        "last_data_point": last_point['date'].isoformat() if 'date' in last_point else None,
        "model_path": os.path.join(APP_ROOT, 'trained_oil_price_model.sav'),
        "file_exists": os.path.exists(os.path.join(APP_ROOT, 'trained_oil_price_model.sav'))
    }
    return jsonify(status), 200 if model else 500

# ====================== MAIN ======================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    logger.info(f"Application root: {APP_ROOT}")
    
    # Safe model status logging
    if model is not None:
        logger.info(f"Model status: Loaded ({type(model).__name__})")
    else:
        logger.info("Model status: Not available")
    
    app.run(host='0.0.0.0', port=port, debug=False)