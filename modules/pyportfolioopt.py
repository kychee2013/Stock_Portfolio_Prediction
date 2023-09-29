import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting

# Calculates expected return over user selected date range i.e. freq (days)
def expected_return(df, freq):
  #expected_returns = []
  #for i in df.columns:
  #  expected_returns.append(df[i].mean()*freq)
  return expected_returns.mean_historical_return(df, frequency=freq)

# Calculates covariance over user selected date range i.e. freq (days)
def covariance(df, freq):
    #return df.cov()*freq
    return risk_models.sample_cov(df, frequency=freq)

# Returns optimal portfolio weightage that maximizes Sharpe Ratio
def weights_max_sharpe(df, freq):
    returns = expected_return(df,freq)
    cov = covariance(df, freq)
    ef = EfficientFrontier(returns, cov, weight_bounds=(-1, 1))
    ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    ef.portfolio_performance(verbose=True)
    return cleaned_weights

# Returns portfolio weightage for user selected return rate
def weights_return_rate(df, freq, return_rate):
    returns = expected_return(df,freq)
    cov = covariance(df, freq)
    ef = EfficientFrontier(returns, cov, weight_bounds=(-1, 1))
    ef.efficient_return(return_rate/100, market_neutral=False)
    cleaned_weights = ef.clean_weights()
    ef.portfolio_performance(verbose=True)
    return cleaned_weights

# Plots the portfolio weights as horizontal bar chart
def plot_weights(weights):
    fig, ax = plt.subplots()
    plotting.plot_weights(weights, ax=ax)

    # Output
    ax.set_title("Portfolio Weightage")
    plt.tight_layout()
    #plt.savefig("ef_scatter.png", dpi=200)
    plt.show()

# Plots the efficient frontier curve **(Work in progress)**
def plot_efficient_frontier(df, freq):
    returns = expected_return(df,freq)
    cov = covariance(df, freq)
    ef = EfficientFrontier(returns, cov, weight_bounds=(-1, 1))
    fig, ax = plt.subplots()
    ef_max_sharpe = ef.deepcopy()
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

    # Find the tangency portfolio
    ef_max_sharpe.max_sharpe()
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

    # Generate random portfolios
    n_samples = 10000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Output
    ax.set_title("Efficient Frontier with random portfolios")
    ax.legend()
    plt.tight_layout()
    #plt.savefig("ef_scatter.png", dpi=200)
    plt.show()
