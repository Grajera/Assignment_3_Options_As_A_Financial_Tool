import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm, lognorm, kstest
from numba import njit, prange

# Constants for stock distribution analysis
BIN_COUNT = 12
ALPHA = 0.5

# Constants for Option Pricing
N_SIMULATIONS = 5000   # Number of price paths to simulate
T = 365                # Time to maturity (365 days)
S0 = 100               # Initial stock price
K = 100                # Strike price of the option
VOLATILITY = 0.1704    # Annual volatility (17.04%)
DRIFT = 0.03           # Drift (3% annual return)
RISK_FREE_RATE = 0.02  # Risk-free rate (2%)
BETA_A, BETA_B = 9, 10 # Beta distribution parameters
SHIFT = 0.35           # Left shift for the beta distribution

def part_one():
    # Load datasets for stock1 and stock2 (no headers, just prices)
    stock1 = pd.read_csv('data/stock1.csv', header=None, names=['Price'])
    stock2 = pd.read_csv('data/stock2.csv', header=None, names=['Price'])

    # Remove non-finite values (NaN, inf) from the stock prices
    prices_stock1 = stock1['Price'].dropna()
    prices_stock2 = stock2['Price'].dropna()

    # Fit a Normal distribution to the stock prices
    mu1, std1 = norm.fit(prices_stock1)
    mu2, std2 = norm.fit(prices_stock2)

    # Fit a Log-Normal distribution to the stock prices
    shape1, loc1, scale1 = lognorm.fit(prices_stock1, floc=0)
    shape2, loc2, scale2 = lognorm.fit(prices_stock2, floc=0)

    # Generate values for plotting the distributions
    x_stock1 = np.linspace(prices_stock1.min(), prices_stock1.max(), 100)
    x_stock2 = np.linspace(prices_stock2.min(), prices_stock2.max(), 100)

    # Plot the histogram of stock1 prices
    plt.figure(figsize=(10, 6))
    plt.hist(prices_stock1, bins=BIN_COUNT, density=True, alpha=ALPHA, color='blue', label='Stock 1 Prices')

    # Plot the Normal distribution fit for stock1 prices
    pdf_norm1 = norm.pdf(x_stock1, mu1, std1)
    plt.plot(x_stock1, pdf_norm1, 'r-', label='Normal Fit for Stock 1')

    # Plot the Log-Normal distribution fit for stock1 prices
    pdf_lognorm1 = lognorm.pdf(x_stock1, shape1, loc1, scale1)
    plt.plot(x_stock1, pdf_lognorm1, 'b-', label='Log-Normal Fit for Stock 1')

    plt.title(f'Distribution Fitting for Stock 1 Prices with {BIN_COUNT} Bins')
    plt.xlabel('Stock Price')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Kolmogorov-Smirnov test for Stock 1
    ks_test_normal1 = kstest(prices_stock1, 'norm', args=(mu1, std1))
    ks_test_lognorm1 = kstest(prices_stock1, 'lognorm', args=(shape1, loc1, scale1))

    print(f"Stock 1 - Normal K-S Test: {ks_test_normal1}")
    print(f"Stock 1 - Log-Normal K-S Test: {ks_test_lognorm1}")

    # Plot the histogram of stock2 prices
    plt.figure(figsize=(10, 6))
    plt.hist(prices_stock2, bins=BIN_COUNT, density=True, alpha=ALPHA, color='red', label='Stock 2 Prices')

    # Plot the Normal distribution fit for stock2 prices
    pdf_norm2 = norm.pdf(x_stock2, mu2, std2)
    plt.plot(x_stock2, pdf_norm2, 'r-', label='Normal Fit for Stock 2')

    # Plot the Log-Normal distribution fit for stock2 prices
    pdf_lognorm2 = lognorm.pdf(x_stock2, shape2, loc2, scale2)
    plt.plot(x_stock2, pdf_lognorm2, 'b-', label='Log-Normal Fit for Stock 2')

    plt.title(f'Distribution Fitting for Stock 2 Prices with {BIN_COUNT} Bins')
    plt.xlabel('Stock Price')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Kolmogorov-Smirnov test for Stock 2
    ks_test_normal2 = kstest(prices_stock2, 'norm', args=(mu2, std2))
    ks_test_lognorm2 = kstest(prices_stock2, 'lognorm', args=(shape2, loc2, scale2))

    print(f"Stock 2 - Normal K-S Test: {ks_test_normal2}")
    print(f"Stock 2 - Log-Normal K-S Test: {ks_test_lognorm2}")

# Monte Carlo Simulation for European Call Option
@njit(parallel=True)
def monte_carlo_simulation(n_simulations, T, S0, K, volatility, drift, risk_free_rate):
    dt = 1 / 365  # Time step (1 day)
    final_stock_prices = np.zeros(n_simulations)  # To store the final stock prices
    payoffs = np.zeros(n_simulations)  # To store the payoff of each simulation

    for i in prange(n_simulations):
        # Initialize stock price path
        stock_price = S0
        
        # Simulate stock price path over time
        for t in range(T):
            Z = np.random.normal()  # Standard normal random variable for the Brownian motion
            stock_price *= np.exp((drift - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * Z)
        
        # Store the final stock price after 365 days
        final_stock_prices[i] = stock_price
        
        # Payoff for a European call option: max(S_T - K, 0)
        payoffs[i] = max(stock_price - K, 0)

    # Average final stock price after 365 days
    average_final_price = np.mean(final_stock_prices)

    # Average payoff (before discounting)
    average_payoff = np.mean(payoffs)

    # Discount the average payoff to present value (this is the cost of the option)
    discounted_payoff = np.exp(-risk_free_rate * T / 365) * average_payoff
    
    return average_final_price, average_payoff, discounted_payoff

# Main function for part_two, which includes the Monte Carlo Simulation
def part_two():
    # Run the Monte Carlo simulation
    avg_final_price, avg_payoff, option_price = monte_carlo_simulation(N_SIMULATIONS, T, S0, K, VOLATILITY, DRIFT, RISK_FREE_RATE)
    
    # Print data out to console for the european option price.
    print(f"Average stock price after 365 days: {avg_final_price:.2f}")
    print(f"Average payoff (before discounting): {avg_payoff:.2f}")
    print(f"Cost of the option (discounted payoff): {option_price:.2f}")
    
    # Plot a sample of 100 price paths
    plt.figure(figsize=(10, 6))
    for _ in range(100):
        stock_price = S0
        path = [stock_price]
        for t in range(T):
            Z = np.random.normal()
            stock_price *= np.exp((DRIFT - 0.5 * VOLATILITY ** 2) * (1 / 365) + VOLATILITY * np.sqrt(1 / 365) * Z)
            path.append(stock_price)
        plt.plot(path, linewidth=0.7, alpha=0.6)
    
    plt.title('Sample Stock Price Paths (Monte Carlo Simulation)')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.show()

    # Run histogram of final stock prices
    final_prices = np.zeros(N_SIMULATIONS)
    for i in range(N_SIMULATIONS):
        stock_price = S0
        for t in range(T):
            Z = np.random.normal()
            stock_price *= np.exp((DRIFT - 0.5 * VOLATILITY ** 2) * (1 / 365) + VOLATILITY * np.sqrt(1 / 365) * Z)
        final_prices[i] = stock_price

    plt.figure(figsize=(10, 6))
    plt.hist(final_prices, bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.axvline(x=K, color='red', linestyle='dashed', linewidth=2, label='Strike Price')
    plt.title('Distribution of Final Stock Prices at Maturity')
    plt.xlabel('Stock Price')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Run part one (stock analysis) and part two (option pricing)
    part_one()
    part_two()
