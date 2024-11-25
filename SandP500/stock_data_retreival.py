import yfinance as yf
import pandas as pd
import numpy as np

# Step 1: Get the list of S&P 500 stocks
def get_sp500_tickers():
    # This is a static list of tickers. Alternatively, scrape or retrieve from an API.
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(url)[0]
    tickers = sp500_table['Symbol'].tolist()
    return tickers

# Step 2: Download historical data
def get_historical_data(tickers, start_date="2015-01-01", end_date="2023-12-31"):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

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
    # Get tickers for S&P 500
    tickers = get_sp500_tickers()
    
    # Limit to a smaller subset for testing (e.g., 50 stocks)
    tickers = tickers[:50]

    # Download historical data
    print("Downloading historical data...")
    price_data = get_historical_data(tickers)
    
    # Drop any stocks with incomplete data
    price_data = price_data.dropna(axis=1)

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
