# oil.py - Updated to ensure datetime compatibility
import pandas as pd 
import yfinance as yf 
import numpy as np 
from statsmodels.tsa.arima.model import ARIMA 
from sklearn.metrics import mean_squared_error 
import joblib 
from datetime import datetime, timedelta
import warnings

def fetch_historical_data():
    """Fetch and preprocess historical oil price data"""
    # Get data until yesterday
    end_date = datetime.now() - timedelta(days=1)
    oil_data = yf.download('CL=F', start='2000-01-01', end=end_date.strftime('%Y-%m-%d'))
    
    # Preprocessing
    df = oil_data[['Close']].copy()
    df['log_close'] = np.log(df['Close'])
    df['log_return'] = df['log_close'].diff()
    df.dropna(inplace=True)
    
    return df

def train_arima_model(data):
    """Train and evaluate ARIMA model"""
    # Split data (80% train, 20% test)
    split_idx = int(len(data) * 0.8)
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]
    
    # Model configuration
    order = (2, 1, 2)  # Determined through grid search
    
    # Train model
    model = ARIMA(train['log_return'], order=order)
    model_fit = model.fit()
    
    # Evaluate
    forecast = model_fit.get_forecast(steps=len(test))
    predicted_returns = forecast.predicted_mean
    actual_returns = test['log_return']
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual_returns, predicted_returns))
    print(f"Model RMSE: {rmse:.6f}")
    
    return model_fit

def main():
    """Main training pipeline"""
    print("Fetching historical oil price data...")
    oil_data = fetch_historical_data()
    
    print("Training ARIMA model...")
    model = train_arima_model(oil_data)
    
    # Save model
    model_path = 'trained_oil_price_model.sav'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save latest data point for reference
    last_point = {
        # Store datetime object directly
        'date': oil_data.index[-1].to_pydatetime(),
        'close': float(oil_data['Close'].iloc[-1]),
        'log_return': float(oil_data['log_return'].iloc[-1])
    }
    joblib.dump(last_point, 'last_data_point.sav')
    
    # Format date for printing
    print(f"Last data point saved: {last_point['date'].strftime('%Y-%m-%d')} - ${last_point['close']:.2f}")

if __name__ == "__main__":
    main()