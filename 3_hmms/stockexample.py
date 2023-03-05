import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import yfinance as ys
import pandas as pd

# INTC = yf.Ticker("INTC")
# INTC.get_shares_full()
# hist = INTC.history(start="1995-01-01", end="2012-01-06")
hist = pd.read_csv("stockprices.csv")
emissions = hist["Close"].to_numpy()

emissions = np.atleast_2d(emissions).T
print(emissions.ndim)

from dynamax.hidden_markov_model import DiagonalGaussianHMM

import jax.numpy as jnp
import jax.random as jr

true_num_states = 4
emission_dim = 1
hmm = DiagonalGaussianHMM(true_num_states, emission_dim)

key = jr.PRNGKey(0)
hmm = DiagonalGaussianHMM(4, emission_dim, transition_matrix_stickiness=10.)
params, props = hmm.initialize(key=key, method="kmeans", emissions=emissions)
params, lps = hmm.fit_em(params, props, emissions, num_iters=100)