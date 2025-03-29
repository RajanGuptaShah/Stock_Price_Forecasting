import os
import time
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from pandas_datareader import data as pdr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from datetime import datetime
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Get user input for stock ticker
TICKER = input("Enter the stock ticker symbol (e.g., AAPL, MSFT, TSLA): ").upper()
START_DATE = "2022-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
CSV_FILE = f"{TICKER}.csv"

def fetch_stock_data(ticker, start_date, end_date, max_retries=5, wait_time=60):
    """
    Fetches stock data from Stooq with volume data.
    """
    for attempt in range(max_retries):
        try:
            print(f"📊 Fetching stock data for {ticker} (Attempt {attempt + 1}/{max_retries})...")
            stock_data = pdr.get_data_stooq(ticker, start=start_date, end=end_date)

            if not stock_data.empty:
                stock_data = stock_data[::-1]  # Reverse order for Prophet
                stock_data.reset_index(inplace=True)
                stock_data = stock_data[['Date', 'Close', 'Volume']]
                stock_data.columns = ['ds', 'y', 'volume']
                stock_data["ds"] = pd.to_datetime(stock_data["ds"]).dt.tz_localize(None)

                # Compute technical indicators
                stock_data['rsi'] = calculate_rsi(stock_data['y'])
                stock_data['ema_9'] = stock_data['y'].ewm(span=9, adjust=False).mean()
                stock_data['ema_21'] = stock_data['y'].ewm(span=21, adjust=False).mean()
                stock_data['macd'], stock_data['macd_signal'] = calculate_macd(stock_data['y'])

                # Fill missing values
                stock_data.fillna(method='ffill', inplace=True)

                print("📊 Data preview (last 5 rows):")
                print(stock_data.tail())  # Print last few rows to check indicators
                print("✅ Data fetched successfully!")
                return stock_data
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            time.sleep(wait_time)
    raise ValueError("❌ Failed to fetch data after multiple attempts.")

def calculate_rsi(series, period=14):
    """
    Calculates the Relative Strength Index (RSI) for a given stock price series.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero issues
    loss.replace(0, np.nan, inplace=True)
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Fill NaN with neutral RSI value

def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    """
    Calculates the MACD and MACD signal line.
    """
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, macd_signal

def forecast_stock_prices(df, periods=30):
    """
    Trains a Prophet model and forecasts stock prices with extra regressors.
    """
    model = Prophet()
    
    # Ensure extra regressors are properly added
    for col in ['volume', 'rsi', 'ema_9', 'ema_21', 'macd', 'macd_signal']:
        if col in df.columns:
            model.add_regressor(col)
        else:
            print(f"⚠️ Warning: Missing regressor {col}")

    model.fit(df)
    
    future = model.make_future_dataframe(periods=periods)
    
    # Ensure future data includes extra regressors
    future = future.merge(df[['ds', 'volume', 'rsi', 'ema_9', 'ema_21', 'macd', 'macd_signal']], on='ds', how='left')
    
    # Fill missing values
    future.fillna(method='ffill', inplace=True)
    
    forecast = model.predict(future)
    return forecast, model

def create_interactive_plot(forecast, df):
    """
    Creates an interactive plot using Plotly.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Prices'))
    
    fig.update_layout(title=f"{TICKER} Stock Price Forecast",
                      xaxis_title='Date',
                      yaxis_title='Stock Price',
                      hovermode='x',
                      template='plotly_dark')
    return fig

def evaluate_model(df, forecast):
    """
    Evaluates model accuracy using MAE, RMSE, and MAPE.
    """
    y_true = df['y'].values
    y_pred = forecast['yhat'][:len(y_true)].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print("\n📊 Model Evaluation Metrics:")
    print(f"✅ Mean Absolute Error (MAE): {mae:.4f}")
    print(f"✅ Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"✅ Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Load existing data or fetch new data
if os.path.exists(CSV_FILE):
    print(f"💾 Loading existing data from {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    df["ds"] = pd.to_datetime(df["ds"])
else:
    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    df.to_csv(CSV_FILE, index=False)
    print(f"💾 Data successfully saved to {CSV_FILE}")

# Train model and forecast
print("📈 Training Prophet model for forecasting...")
forecast, model = forecast_stock_prices(df, periods=30)

# Create interactive web page
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1(f"{TICKER} Stock Forecast Dashboard"),
    dcc.Graph(id='stock-forecast', figure=create_interactive_plot(forecast, df)),
    dcc.Slider(
        id='forecast-slider',
        min=10,
        max=90,
        step=10,
        value=30,
        marks={i: f"{i} days" for i in range(10, 100, 10)}
    )
])

@app.callback(
    Output('stock-forecast', 'figure'),
    Input('forecast-slider', 'value')
)
def update_forecast(periods):
    new_forecast, _ = forecast_stock_prices(df, periods=periods)
    return create_interactive_plot(new_forecast, df)

if __name__ == '__main__':
    app.run(debug=False)
