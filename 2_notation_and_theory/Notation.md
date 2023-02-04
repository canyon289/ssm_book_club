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

# How I Learn Notation


## Why Notation


1. It's the most compact and precise way to represent math
2. There's no way to avoid it with the texts we've picked


## Full State Space Model Equation


$$p(y_{1:T}, z_{1:T} | u_{1:T}) = p(z_1 | u_1) p(y_1 | z_1, u_1) \prod_{t=1}^T p(z_t | z_{t-1}, u_t) p(y_t | z_t, u_t)$$


ProbML Equation 29.5


## Bayes Theorem


$$ \underbrace{p(\boldsymbol{\theta} \mid \boldsymbol{Y})}_{\text{posterior}} = \frac{\overbrace{p(\boldsymbol{Y} \mid \boldsymbol{\theta})}^{\text{likelihood}}; \overbrace{p(\boldsymbol{\theta})}^{\text{prior}}}{\underbrace{{{\int_{\boldsymbol{\Theta}} p(\boldsymbol{Y} \mid \boldsymbol{\theta})p(\boldsymbol{\theta}) d\boldsymbol{\theta}}}}_{\text{marginal likelihood}}} $$


## My steps


1. Understand the fundamental philosophy
2. Find the simplest applied example
  * Typically in code
3. Implement it myself
4. Add complexity one step at a time


This is what we did just did for Bayes Theorem


## Bayes Theorem Simplified


Started here


$$
\text{Posterior} = \frac{\text{Likelihood} ; * \text{Prior}}{\text{Marginal-Likelihood}}
$$


From there we built back up to Linear Regression


## Things I looks for


1. What type of math am I dealing with?
  * Probability
  * Distributions
  * Integrals etc
2. Figure out what's a scalar, vector, matrix
  * Figure out whats a real number, indicator, random variable
3. Pay attention to shapes


## State Space Model Simplified Version


$$
\begin{align}
p(y_{1:T}, z_{1:T} \mid \theta) 
&= \mathrm{Cat}(z_1 \mid \pi) 
\prod_{t=2}^T \mathrm{Cat}(z_t \mid A_{z_{t-1}}) 
\prod_{t=1}^T \mathrm{Cat}(y_t \mid B_{z_t})
\end{align}
$$


## Recap
* Notation can be challenging but is important
* Break it down one step at a time
* Learn the meaning not just the symbols
* Building things in code really help me

