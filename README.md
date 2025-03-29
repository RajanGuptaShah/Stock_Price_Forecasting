# Stock Price Forecasting

## Overview
Stock Price Forecasting is a data-driven project that predicts stock market trends using advanced machine learning techniques. This project utilizes historical stock data to analyze market patterns and make future price predictions.

## Features
- Fetches stock data using Stooq API
- Computes technical indicators like RSI, EMA, and MACD
- Time series forecasting using Facebook Prophet
- Visualization of stock trends using Plotly
- Model evaluation with MAE, RMSE, and MAPE metrics
- Interactive web dashboard using Dash
- **Offline Mode:** If the server is not working, the system automatically downloads the CSV file and processes the data locally.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: pandas, numpy, plotly, scikit-learn, Prophet, Dash
- **Frameworks**: Machine Learning, Time Series Analysis
- **Data Visualization**: Plotly, Dash

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/RajanGuptaShah/Stock_Price_Forecasting.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Stock_Price_Forecasting
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the main script to start stock forecasting:
   ```bash
   python stock_forecast.py
   ```
2. Enter the stock ticker when prompted (e.g., AAPL, MSFT, TSLA).
3. The script will fetch stock data, train a Prophet model, and generate forecasts.
4. To view interactive visualizations, open the Dash web dashboard:
   ```bash
   python app.py
   ```

## Dataset
The dataset consists of historical stock market data, including:
- Date (`ds`)
- Closing price (`y`)
- Volume
- Relative Strength Index (RSI)
- Exponential Moving Averages (EMA 9, EMA 21)
- Moving Average Convergence Divergence (MACD, MACD Signal)

## Results
The model is evaluated based on:
- **Mean Absolute Error (MAE)**: Measures prediction accuracy
- **Root Mean Square Error (RMSE)**: Assesses error magnitude
- **Mean Absolute Percentage Error (MAPE)**: Measures percentage error

## Future Enhancements
- Incorporate sentiment analysis using financial news
- Improve model accuracy with hybrid machine learning techniques
- Develop a more user-friendly web-based dashboard for real-time predictions

## Contributing
Contributions are welcome! If you'd like to improve this project, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For any queries or suggestions, please contact [Rajan Kumar Gupta](https://github.com/RajanGuptaShah).

