# Assignment_3_Options_As_A_Financial_Tool

Group Members:
- TODO: Add your name here
- Chase Mortensen

## Running the program

In order to run the program, install the following packages: `pandas`, `matplotlib`, `numpy`, and `scipy`.

Then, run

```sh
python main.py
```

You may comment out specific sections in the `main` function.

## Part 1: Fitting Stock Data to Distributions

__Task:__ Provide a plot showing the stock data overlaid with the fitted distributions.

![image info](./img/stock_1_hist_12_bins.png)
<p align="center">Fig. 1. Stock 1 with Normal and Log-Normal Distributions overlaid over a histogram with 12 bins.</p>

![image info](./img/stock_2_hist_12_bins.png)
<p align="center">Fig. 2. Stock 2 with Normal and Log-Normal Distributions overlaid over a histogram with 12 bins.</p>

Note: More graphs are included in the `img` folder that include the same distributions with varying bins. 12 bins seems to show the distribution but the other graphs are also helpful.

__Task:__ Report on the best-fitting distributions for both stocks and the metrics used to measure the goodness of fit.

First, visually comparing the Normal and Log-Normal distributions for each graph reveals a slight right-skew on both of the Log-Normal overlays, which seems to fit the data better in both instances. The peak of the Log-Normal distributions in both cases also seems to be slightly less than the Normal distributions.

The Kolmogorov-Smirnov test gives us the following data:

### Stock 1
- Normal Distribution
  - K-S Statistic: 0.032
  - p-value 0.832
- Log-Normal Distribution
  - K-S Statistic: 0.025
  - p-value 0.973

The Log-Normal has a lower K-S statistic and higher p-value, which signifies a better fit for the Log-Normal distribution for Stock 1.

### Stock 2
- Normal Distribution
  - K-S Statistic: 0.036
  - p-value 0.719
- Log-Normal Distribution
  - K-S Statistic: 0.025
  - p-value 0.974

Again, the Log-Normal distribution has a lower K-S statistic and higher p-value than the Normal distribution, so it is also a better fit for the Stock 2 data.