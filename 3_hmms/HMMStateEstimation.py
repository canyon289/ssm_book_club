
from functools import partial

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import vmap
from jax.nn import one_hot

from dynamax.hidden_markov_model import CategoricalHMM


initial_probs = jnp.array([0.5, 0.5])
transition_matrix = jnp.array([[0.95, 0.05],
                               [0.10, 0.90]])

# transition_matrix = jnp.array([[1.0, 0.0],
#                                [0.0, 1.0]])

emission_probs = jnp.array([[1/6,  1/6,  1/6,  1/6,  1/6,  1/6],    # fair die
                            [1/10, 1/10, 1/10, 1/10, 1/10, 5/10]])  # loaded die

emission_probs = jnp.array([[1/2,  1/2],    # fair die
                            [1/10, 9/10]])  # loaded die

print(f"A.shape: {transition_matrix.shape}")
print(f"B.shape: {emission_probs.shape}")

num_states = 2      # two types of dice (fair and loaded)
num_emissions = 1   # only one die is rolled at a time
num_classes = 2     # each die has six faces

# Construct the HMM
hmm = CategoricalHMM(num_states, num_emissions, num_classes)

# Initialize the parameters struct with known values
params, _ = hmm.initialize(initial_probs=initial_probs,
                           transition_matrix=transition_matrix,
                           emission_probs=emission_probs.reshape(num_states, num_emissions, num_classes))

num_timesteps =3
true_states, emissions = hmm.sample(params, jr.PRNGKey(42), num_timesteps)
print(true_states, emissions)
# print(f"true_states.shape: {true_states.shape}")
# print(f"emissions.shape: {emissions.shape}")
# print("")
# print("First few states:    ", true_states[:5])
# print("First few emissions: ", emissions[:5, 0])

# To sample multiple sequences, just use vmap
# In this case 5 sequences were modeled
num_batches = 5

batch_states, batch_emissions = \
    vmap(partial(hmm.sample, params, num_timesteps=num_timesteps))(
        jr.split(jr.PRNGKey(0), num_batches))

# print(f"batch_states.shape: {batch_states.shape}")
# print(f"batch_emissions.shape: {batch_emissions.shape}")

# count fraction of times we see 6 in each state
# remember that python is zero-indexed, so we have to add one!
p0 = jnp.mean(emissions[true_states==0] + 1 == 6)   # fair
p1 = jnp.mean(emissions[true_states==1] + 1 == 6)   # loaded
# print("empirical frequencies: ", jnp.array([p0, p1]))
# print("expected frequencies:  ", emission_probs[:, -1])

posterior = hmm.filter(params, emissions)
# print(f"marginal likelihood: {posterior.marginal_loglik: .2f}")
# print(f"posterior.filtered_probs.shape: {posterior.filtered_probs.shape}")


# Manual first step calcu
# Probability of being on the loaded state