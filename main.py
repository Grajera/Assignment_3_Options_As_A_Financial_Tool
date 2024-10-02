import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, lognorm, kstest

BIN_COUNT = 12
ALPHA = 0.5


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


if __name__ == "__main__":

    # comment out specific sections if desired
    part_one()
