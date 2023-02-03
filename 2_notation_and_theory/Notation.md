---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

## How I Learn Notation


1. Understand the fundamental philosophy
2. Find the simplest applied example
  * Typically in code
3. Implement it myself
4. Add complexity one step at a time


## Things I looks for


1. What type of math am I dealing with?
  * Probsbility
  * Distributions
  * Integrals etc
2. Figure out what's a scalar, vector, matrix
  * Figure out whats a real number, indicator, random variablle
3. Pay attention to shapes


## Full State Space Model Equation


$$p(y_{1:T}, z_{1:T} | u_{1:T}) = p(z_1 | u_1) p(y_1 | z_1, u_1) \prod_{t=1}^T p(z_t | z_{t-1}, u_t) p(y_t | z_t, u_t)$$


## Simplified Version


\begin{align}
p(y_{1:T}, z_{1:T} \mid \theta) 
&= \mathrm{Cat}(z_1 \mid \pi) 
\prod_{t=2}^T \mathrm{Cat}(z_t \mid A_{z_{t-1}}) 
\prod_{t=1}^T \mathrm{Cat}(y_t \mid B_{z_t})
\end{align}


## State Space Models


Things I need to explain
* State - The hidden thing we can't observe
  * Transition model
* Space - Where we see outcomes or emissions
  * Emission model
* Outcome or emissions


## Set of hidden states

```python

```

## Time steps

```python

```

## Observations at those time steps


## Notation
$$\theta = (\pi, A, B)$$

A - Transition Matrix
B - Emission Probability
$\pi$ - Initial probability


What I can do here is build up the diagram one piece at a time to talk through initial state, state, space


## The Task


If we see a bunch of outcomes can we
* Estimate what state we are in, or were in, or will be in
* Estimate the transition matrix
  * If we don't know it
* Estimate the starting point
  * Less useful most of the time but 


## Estimators
for point estimates
* Stochastic Gradient Descent
* Expectation Maximization

For full posteriors
  * HMC


## Next Time
* Casino HMM
* Filtering (forwards algorithm)
* Smoothing (forwards-backwards algorithm)
* Most likely state sequence (Viterbi algorithm)
