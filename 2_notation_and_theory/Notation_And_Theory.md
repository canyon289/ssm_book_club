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

# State Space Model Book Club


## Agenda


1. Bayesian Update Intuition
  * Classic Coin Flips (and now Covid)
  * AB Testing
  
2. How I learn notation


## Bayes Formula


$$ \underbrace{p(\boldsymbol{\theta} \mid \boldsymbol{Y})}_{\text{posterior}} = \frac{\overbrace{p(\boldsymbol{Y} \mid \boldsymbol{\theta})}^{\text{likelihood}}; \overbrace{p(\boldsymbol{\theta})}^{\text{prior}}}{\underbrace{{{\int_{\boldsymbol{\Theta}} p(\boldsymbol{Y} \mid \boldsymbol{\theta})p(\boldsymbol{\theta}) d\boldsymbol{\theta}}}}_{\text{marginal likelihood}}} $$


## Bayesian Update


1. You have some prior belief
  * It may be opinionated/informmed
  * It may not


2. You get some data



3. You update your beliefs


## Simplified Bayes Formula


$$
\text{Posterior} = \frac{\text{Likelihood} ; * \text{Prior}}{\text{Marginal-Likelihood}}
$$


## The typical problems


* Coin Flips
* COVID 
* Monty Hall Problem

<!-- #region slideshow={"slide_type": "slide"} -->
<center>
  <img src="img/ProbMLCOVID.png" style="height:850px"; />
</center>
<!-- #endregion -->

## Covid Prevalance


$$
P(SF) = \text{10%} \\
P(\text{~}SF) = \text{90%}
$$


## Likelihood of posistive test


$$
P(PT | SF) = \text{Chance of a positive test given the person has space-flu.} \\
P(PT | \text{~}SF) = \text{Chance of a positive test given the person doesn’t have space flu.}
$$


$$
P(PT \mid SF) = \text{90%} \\
P(PT \mid \text{~}SF) = \text{20%}
$$


## Covid in code

```python
prior = [.1, .9]  # P(SF), P(~SF)
likelihood = [.9, .2]  # P(PT | SF), p(PT | ~SF)
```

```python
unnormalized_posterior = [None, None]
unnormalized_posterior[0] = likelihood[0]*prior[0]
unnormalized_posterior[1] = likelihood[1]*prior[1]
```

```python
marginal_likelihood = likelihood[0]*prior[0] + likelihood[1]*prior[1]
```

```python
posterior = [None, None]
posterior[0] = unnormalized_posterior[0] / marginal_likelihood
posterior[1] = unnormalized_posterior[1] / marginal_likelihood  
posterior
```

## COVID Example Visualized


**Insert Upload to Youtube here**


## Inverse Problems


<center>
  <img src="img/InverseProblems.png" style="height:850px"; />
</center>


## We see something, what do we learn?
Not, we know something (probability values) what is the probability of some subevent occurring?


## User Conversion Probability on a website?


* A 100 visitors visit us
* 8 Convert

What is the conversion rate?


## What are **possible conversion rates**


Possible conversion rates

* 0%
* 8%
* 20%
* 31%
* 88%
* 99%


All are possible, except 0%


## What is the plausibility of the conversion rates?


### Lets start with priors

```python
pz.Beta(2, 2).plot_pdf(figsize=(24,12));
```

## Bayesian Update

```python
num_conversions = 8
num_non_conversions = 100 - num_conversions
pz.Beta(2+8, 2+num_non_conversions).plot_pdf();
```

## Relatively Plausibility of all possible beliefs

```python
num_conversions = 8
num_non_conversions = 100 - num_conversions
pz.Beta(2+8, 2+num_non_conversions).plot_pdf();
```

## Bayesian Update with a PPL

```python
with pm.Model() as model:
    θ = pm.Beta("θ", 2, 10)
    y = pm.Binomial("y", n=1, p=θ, observed=signups)
    trace = pm,sample()
```




## What's the difference



* Conjugate Model - Pure "pen on paper math"
  * No computer needed
  * Exact
  * Very restricted to specific prior likelihood combinations


* Markov Chain Monte Carlo algorithms
  * Not very practical without computers
  * Enables 
  * Generally applicable


## Dynamax Book Club Takeaway


* Various SSMs have "pen and paper" solutions
* With tools like Dyna


We want to learn both the traditional techniques **and** what newer tools like JAX and Dynamax let us solve








## Bayesian Intution Recap


* Bayes theorem is a philosophy for how we can update our beliefs given observations
* ProbML calls outcome -> belief mapping inverse probability
* What we care about is the relative plausibiilty
