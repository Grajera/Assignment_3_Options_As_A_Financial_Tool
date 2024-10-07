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

    print(f"\nPart 1: Fitting Stock Data to Distributions")
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
    print(f"\nPart 2: Monte Carlo Simulation for Vanilla European Option Pricing")
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

#Monte Carlo with Basket Option
@njit(parallel=True)
def monte_carlo_basket_option(n_simulations, T, S0_1, S0_2, volatility_1, volatility_2, drift_1, drift_2, K, risk_free_rate):
    dt = 1 / 365  # Time step (1 day)
    final_prices_1 = np.zeros(n_simulations)
    final_prices_2 = np.zeros(n_simulations)
    payoffs_scenario_1 = np.zeros(n_simulations)
    payoffs_scenario_2 = np.zeros(n_simulations)

    for i in prange(n_simulations):
        stock_price_1 = S0_1
        stock_price_2 = S0_2

        # Simulate both stock price paths over time
        for t in range(T):
            Z1 = np.random.normal()  # Standard normal random variable for stock1
            Z2 = np.random.normal()  # Standard normal random variable for stock2
            stock_price_1 *= np.exp((drift_1 - 0.5 * volatility_1 ** 2) * dt + volatility_1 * np.sqrt(dt) * Z1)
            stock_price_2 *= np.exp((drift_2 - 0.5 * volatility_2 ** 2) * dt + volatility_2 * np.sqrt(dt) * Z2)

        # Store the final stock prices after 365 days
        final_prices_1[i] = stock_price_1
        final_prices_2[i] = stock_price_2

        # Scenario 1: Payoff based on the average of both stock prices at maturity
        average_price = (stock_price_1 + stock_price_2) / 2
        payoffs_scenario_1[i] = max(average_price - K, 0)

        # Scenario 2: Payoff based on the maximum of both stock prices at maturity
        max_price = max(stock_price_1, stock_price_2)
        payoffs_scenario_2[i] = max(max_price - K, 0)

    # Discount the average payoffs to present value (cost of the option)
    discounted_payoff_scenario_1 = np.exp(-risk_free_rate * T / 365) * np.mean(payoffs_scenario_1)
    discounted_payoff_scenario_2 = np.exp(-risk_free_rate * T / 365) * np.mean(payoffs_scenario_2)

    return discounted_payoff_scenario_1, discounted_payoff_scenario_2

# Main function for part_three, which includes the Monte Carlo Simulation with Basket Option
def part_three():
    # Load datasets for stock1 and stock2 (no headers, just prices)
    stock1 = pd.read_csv('data/stock1.csv', header=None, names=['Price'])
    stock2 = pd.read_csv('data/stock2.csv', header=None, names=['Price'])

    # Remove non-finite values (NaN, inf) from the stock prices
    prices_stock1 = stock1['Price'].dropna()
    prices_stock2 = stock2['Price'].dropna()

    # Fit a Log-Normal distribution to the stock prices (best-fitting distribution from Part 1)
    shape1, loc1, scale1 = lognorm.fit(prices_stock1, floc=0)
    shape2, loc2, scale2 = lognorm.fit(prices_stock2, floc=0)

    # Run the Monte Carlo simulation for both stocks
    option_price_scenario_1, option_price_scenario_2 = monte_carlo_basket_option(
        N_SIMULATIONS, T, S0, S0, VOLATILITY, VOLATILITY, DRIFT, DRIFT, K, RISK_FREE_RATE)

    # Print the results for both scenarios
    print(f"\nPart 3: Stochastic Jumps and Basket Option Pricing")
    print(f"Option price for Average value: {option_price_scenario_1:.2f}")
    print(f"Option price for Maximum value: {option_price_scenario_2:.2f}")

    # Plot a histogram of the final stock prices for both stocks (from Monte Carlo)
    final_prices_1 = np.zeros(N_SIMULATIONS)
    final_prices_2 = np.zeros(N_SIMULATIONS)

    for i in range(N_SIMULATIONS):
        stock_price_1 = S0
        stock_price_2 = S0
        for t in range(T):
            Z1 = np.random.normal()
            Z2 = np.random.normal()
            stock_price_1 *= np.exp((DRIFT - 0.5 * VOLATILITY ** 2) * (1 / 365) + VOLATILITY * np.sqrt(1 / 365) * Z1)
            stock_price_2 *= np.exp((DRIFT - 0.5 * VOLATILITY ** 2) * (1 / 365) + VOLATILITY * np.sqrt(1 / 365) * Z2)
        final_prices_1[i] = stock_price_1
        final_prices_2[i] = stock_price_2

    # Plot histograms for the final stock prices at maturity
    plt.figure(figsize=(10, 6))
    plt.hist(final_prices_1, bins=50, alpha=0.7, color='blue', label='Final Stock 1 Prices')
    plt.hist(final_prices_2, bins=50, alpha=0.7, color='green', label='Final Stock 2 Prices')
    plt.axvline(x=K, color='red', linestyle='dashed', linewidth=2, label='Strike Price')
    plt.title('Distribution of Final Stock Prices at Maturity (Best Fit Distribution vs Monte Carlo with Basket Options)')
    plt.xlabel('Stock Price')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Run part one (stock analysis), part two (option pricing), and part three (basket option pricing)
    part_one()
    part_two()
    part_three()
