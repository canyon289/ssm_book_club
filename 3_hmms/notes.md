# General Notes

1. HMM Evidence Likelihood
2. Forward algorithm - Used to calculate the probability were in a state. Assume we know prior, transition, and emission probabilities
3. Backward algorithm
4. Viterbi algorithm
5. Parameter learning

## HMM Evidence Likelihood
* The website uses this notation $L(X \vert \theta)$
* By the time we get here $P(HHH \vert z_{3}=F) = \Big[P(HH \vert z_{2}=F)*A_{FF}+P(HH \vert z_{2}=B)*A_{BF}\Big] \phi(H \vert F)$  we only care about two things.
  1. The probability of the sequences that got us the last two heads
  2. The emission probability at that time
* The thing is to calculate the probability of how we got those two heads we recurse down into that probability

* $\alpha_t(i) = P(X_{1:t}\vert z_{t}=i)$ is the probability of being in state 'i' at the time 't' given the 'observations till time t'.
* We can figure this out by summing a bunch of branches
  * I think dot means dot product?


# Book CLub
* Understand the point of SSMs like filtering etc
* Be able to name the algorithms
* Identify them in the Dynamax codebase
* Detail specific things like dirichlet priors

# TODO
* Simplify and run an HMM notebook and see what happens
* What is prior concentration and why is it set to 1.1 https://probml.github.io/dynamax/api.html#dynamax.hidden_markov_model.CategoricalHMM


## State Estimation Filtering

### Instantiate an HMM class with initial parameters
https://github.com/probml/dynamax/blob/9650ee9940f229f6ea8349cb6288a3f4f41fec02/dynamax/hidden_markov_model/models/categorical_hmm.py#L30

Those parameters are
   * Initial probability
   * Transition Matrix
   * Emission probabilities
   * Prior Concentration (This one is optional)
    
In this model we're not inferring or estimating any of these paramters.
It is important to note though that the initial probability is 
mutated a bit under the hood with a dirichlet prior.

### Call initialize
https://github.com/probml/dynamax/blob/9650ee9940f229f6ea8349cb6288a3f4f41fec02/dynamax/hidden_markov_model/models/categorical_hmm.py#L59

This constructs the priors from the MLE provided.
Currently implements Dirichlet for sampling
This then samples from the prior to get the values.

## Then calls filter
This is what I need to learn.
* Get transition matrix
* The log likelihoods seem to be time varying
  * Figure out how these are calculated
* Learn how condition on is implemented
* Predict

* Read more on JAX jit
  * https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html

* Do the first couple by hand to really get an idea of whats going on
  * This will be a great use of the book club
* There's a lot of probabilities going on
  * Emissions, transition, state
  * In this filtering function we are trying to estimate prob of state


## Book Club
1. Talk about why this is confusing
  * So much notation
  * So many probabilities, and likelihoods
2. Three algorithms for state estimation
  * Multiple names smoothing, Forward Backward algorithm
3. Live example
  * How I go through programs
  * Calculating filter and predictive steps
4. Real world use case
5. Parameter estimation
6. 