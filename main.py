import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, lognorm, beta

BIN_COUNT = 30
ALPHA = 0.5
EPSILON = 1e-5  # Small value to ensure the data stays within (0, 1)

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
shape1, loc1, scale1 = lognorm.fit(prices_stock1, floc=0)  # Fix loc=0 for log-normal fitting
shape2, loc2, scale2 = lognorm.fit(prices_stock2, floc=0)

# Normalize stock prices to fit the Beta distribution (Beta is defined on [0, 1])
prices_stock1_normalized = (prices_stock1 - prices_stock1.min()) / (prices_stock1.max() - prices_stock1.min())
prices_stock2_normalized = (prices_stock2 - prices_stock2.min()) / (prices_stock2.max() - prices_stock2.min())

# Shift normalized data slightly away from the boundaries 0 and 1
prices_stock1_normalized = np.clip(prices_stock1_normalized, EPSILON, 1 - EPSILON)
prices_stock2_normalized = np.clip(prices_stock2_normalized, EPSILON, 1 - EPSILON)

# Fit a Beta distribution to the normalized stock prices
a1, b1, loc_beta1, scale_beta1 = beta.fit(prices_stock1_normalized, floc=0, fscale=1)
a2, b2, loc_beta2, scale_beta2 = beta.fit(prices_stock2_normalized, floc=0, fscale=1)

# Generate values for plotting the distributions
x_stock1 = np.linspace(prices_stock1.min(), prices_stock1.max(), 100)
x_stock2 = np.linspace(prices_stock2.min(), prices_stock2.max(), 100)

# For beta, we need normalized x values
x_stock1_normalized = np.linspace(0, 1, 100)
x_stock2_normalized = np.linspace(0, 1, 100)

# Plot the histogram of stock1 prices
plt.figure(figsize=(10, 6))
plt.hist(prices_stock1, bins=BIN_COUNT, density=True, alpha=ALPHA, color='blue', label='Stock 1 Prices')

# Plot the Normal distribution fit for stock1 prices
pdf_norm1 = norm.pdf(x_stock1, mu1, std1)
plt.plot(x_stock1, pdf_norm1, 'r-', label='Normal Fit for Stock 1')

# Plot the Log-Normal distribution fit for stock1 prices
pdf_lognorm1 = lognorm.pdf(x_stock1, shape1, loc1, scale1)
plt.plot(x_stock1, pdf_lognorm1, 'b-', label='Log-Normal Fit for Stock 1')

# # Plot the Beta distribution fit for stock1 prices (scaled back to original range)
# pdf_beta1 = beta.pdf(x_stock1_normalized, a1, b1, loc=loc_beta1, scale=scale_beta1)
# plt.plot(x_stock1, pdf_beta1, 'g-', label='Beta Fit for Stock 1 (Normalized)')

plt.title('Distribution Fitting for Stock 1 Prices')
plt.xlabel('Stock Price')
plt.ylabel('Density')
plt.legend()
plt.show()

# Plot the histogram of stock2 prices
plt.figure(figsize=(10, 6))
plt.hist(prices_stock2, bins=BIN_COUNT, density=True, alpha=ALPHA, color='red', label='Stock 2 Prices')

# Plot the Normal distribution fit for stock2 prices
pdf_norm2 = norm.pdf(x_stock2, mu2, std2)
plt.plot(x_stock2, pdf_norm2, 'r-', label='Normal Fit for Stock 2')

# Plot the Log-Normal distribution fit for stock2 prices
pdf_lognorm2 = lognorm.pdf(x_stock2, shape2, loc2, scale2)
plt.plot(x_stock2, pdf_lognorm2, 'b-', label='Log-Normal Fit for Stock 2')

# # Plot the Beta distribution fit for stock2 prices (scaled back to original range)
# pdf_beta2 = beta.pdf(x_stock2_normalized, a2, b2, loc=loc_beta2, scale=scale_beta2)
# plt.plot(x_stock2, pdf_beta2, 'g-', label='Beta Fit for Stock 2 (Normalized)')

plt.title('Distribution Fitting for Stock 2 Prices')
plt.xlabel('Stock Price')
plt.ylabel('Density')
plt.legend()
plt.show()
