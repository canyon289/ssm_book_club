
from functools import partial

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import vmap
from jax.nn import one_hot

from dynamax.hidden_markov_model import CategoricalHMM


initial_probs = jnp.array([0.5, 0.5])

transition_matrix = jnp.array([[1.0, 0.0],
                               [0.0, 1.0]])

# transition_matrix = jnp.array([[.5, .5],
#                                [0.0, 1.0]])

emission_probs = jnp.array([[1/2,  1/2],    # fair die
                            [1/10, 9/10]])  # loaded die

num_states = 2      # two types of dice (fair and loaded)
num_emissions = 1
num_classes = 2

# Construct the HMM
hmm = CategoricalHMM(num_states, num_emissions, num_classes)

# Initialize the parameters struct with known values
params, _ = hmm.initialize(initial_probs=initial_probs,
                           transition_matrix=transition_matrix,
                           emission_probs=emission_probs.reshape(num_states, num_emissions, num_classes))

num_timesteps =3
true_states, emissions = hmm.sample(params, jr.PRNGKey(42), num_timesteps)
print(true_states, emissions)

num_batches = 5

batch_states, batch_emissions = \
    vmap(partial(hmm.sample, params, num_timesteps=num_timesteps))(
        jr.split(jr.PRNGKey(0), num_batches))

p0 = jnp.mean(emissions[true_states==0] + 1 == 6)   # fair
p1 = jnp.mean(emissions[true_states==1] + 1 == 6)   # loaded

posterior = hmm.filter(params, emissions)