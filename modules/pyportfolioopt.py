import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

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
