import requests
import pandas as pd
import yfinance as yf
import numpy as np

# Step 1: Get the list of NIFTY 50 stocks dynamically from NSE API
def get_nifty50_tickers():
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/"
    }
    
    # Start a session to maintain cookies
    session = requests.Session()
    session.headers.update(headers)
    
    # Initial request to the NSE homepage (to obtain cookies)
    session.get("https://www.nseindia.com")
    
    # Fetch the data from the API
    response = session.get(url)
    if response.status_code == 401:
        raise Exception("Unauthorized access. Check if NSE has updated its access policies.")
    response.raise_for_status()  # Ensure the request was successful

    # Parse JSON data
    data = response.json()
    ticker_data = data['data']
    
    # Extract ticker symbols and append `.NS` for Yahoo Finance
    tickers = [stock['symbol'] + ".NS" for stock in ticker_data]
    return tickers

# Step 2: Download historical data


# Step 3: Calculate returns
def calculate_returns(price_data):
    returns = price_data.pct_change().dropna()
    return returns

# Step 4: Compute covariance matrix and expected returns
def compute_statistics(returns):
    covariance_matrix = returns.cov()
    expected_returns = returns.mean() * 252  # Annualized returns
    return covariance_matrix, expected_returns

# Main Execution
if __name__ == "__main__":
    # Get tickers for NIFTY 50
    print("Fetching NIFTY 50 tickers...")
    tickers = get_nifty50_tickers()
    
    # Limit to a smaller subset for testing (e.g., 50 stocks)
    # tickers = tickers[:50]

    # Download historical data
    print("Downloading historical data...")
    price_data = yf.download(tickers, start="2015-01-01", end="2023-12-31")['Adj Close']
    
    price_data = price_data.dropna(axis=1) # Drop columns with NaN values

    # Calculate daily returns
    print("Calculating returns...")
    returns = calculate_returns(price_data)
    
    # Compute covariance matrix and expected returns
    print("Computing statistics...")
    covariance_matrix, expected_returns = compute_statistics(returns)
    
    # Save results to CSV files
    price_data.to_csv('price_data.csv')
    returns.to_csv('returns.csv')
    covariance_matrix.to_csv('covariance_matrix.csv')
    expected_returns.to_csv('expected_returns.csv', header=True)
    
    print("Data saved to CSV files.")
