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

<!-- #region slideshow={"slide_type": "slide"} -->
# How I Learn Notation
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Why Notation
<!-- #endregion -->

1. It's the most compact and precise way to represent math
2. There's no way to avoid it with the texts we've picked

<!-- #region slideshow={"slide_type": "slide"} -->
## Full State Space Model Equation
<!-- #endregion -->

$$p(y_{1:T}, z_{1:T} | u_{1:T}) = p(z_1 | u_1) p(y_1 | z_1, u_1) \prod_{t=1}^T p(z_t | z_{t-1}, u_t) p(y_t | z_t, u_t)$$


ProbML Equation 29.5

<!-- #region slideshow={"slide_type": "slide"} -->
## Bayes Theorem
<!-- #endregion -->

$$ \underbrace{p(\boldsymbol{\theta} \mid \boldsymbol{Y})}_{\text{posterior}} = \frac{\overbrace{p(\boldsymbol{Y} \mid \boldsymbol{\theta})}^{\text{likelihood}}; \overbrace{p(\boldsymbol{\theta})}^{\text{prior}}}{\underbrace{{{\int_{\boldsymbol{\Theta}} p(\boldsymbol{Y} \mid \boldsymbol{\theta})p(\boldsymbol{\theta}) d\boldsymbol{\theta}}}}_{\text{marginal likelihood}}} $$

<!-- #region slideshow={"slide_type": "slide"} -->
## My steps
<!-- #endregion -->

1. Understand the fundamental philosophy
2. Find the simplest applied example
  * Typically in code
3. Implement it myself
4. Add complexity one step at a time

<!-- #region slideshow={"slide_type": "fragment"} -->
This is what we did just did for Bayes Theorem
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Bayes Theorem Simplified
<!-- #endregion -->

$$
\text{Posterior} = \frac{\text{Likelihood} ; * \text{Prior}}{\text{Marginal-Likelihood}}
$$

<!-- #region slideshow={"slide_type": "fragment"} -->
From there we built back up to Linear Regression
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Things I looks for
<!-- #endregion -->

1. What type of math am I dealing with?
  * Probability
  * Distributions
  * Integrals etc
2. Figure out what's a scalar, vector, matrix
  * Figure out whats a real number, indicator, random variable
3. Pay attention to shapes

<!-- #region slideshow={"slide_type": "slide"} -->
## State Space Model Simplified Version
<!-- #endregion -->

$$
\begin{align}
p(y_{1:T}, z_{1:T} \mid \theta) 
&= \mathrm{Cat}(z_1 \mid \pi) 
\prod_{t=2}^T \mathrm{Cat}(z_t \mid A_{z_{t-1}}) 
\prod_{t=1}^T \mathrm{Cat}(y_t \mid B_{z_t})
\end{align}
$$

<!-- #region slideshow={"slide_type": "slide"} -->
## What I suggest for you
<!-- #endregion -->

1. Develop a strategy that works for you
  * Use mine, or find the combination that works for you
2. Utilize code, examples, notecards whatever
  * Use the SSM community
3. Don't get frustrated, you can do it
  * It's like learning a new language

<!-- #region slideshow={"slide_type": "slide"} -->
## Recap
* Notation can be challenging but is important
* Break it down one step at a time
* Learn the meaning not just the symbols
* Building things in code really help me

<!-- #endregion -->
