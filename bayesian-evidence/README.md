# Bayesian Evidence Playground

A compact, reproducible sandbox for **Bayesian linear regression model-selection**.  
You’ll compute the log-evidence of competing polynomial basis models, visualise how evidence changes with model complexity and data noise, and see how evidence guides you to the “best” (and worst!) model.

---

## What we do — step by step

1. **Implement the log-evidence formula** for a Bayesian linear-regression model  
   $$
   \[
   \log p \!\left(y \mid \mu_\theta,\Sigma_\theta,\sigma^2\right)
   = \tfrac12\log\frac{\lvert\Sigma_\theta\!\mid_{\!\mathcal D}}{\lvert\Sigma_\theta\rvert}
   -\tfrac12\Bigl[(\mu_{\theta\mid\mathcal D}-\mu_\theta)^\top\Sigma_\theta^{-1}(\mu_{\theta\mid\mathcal D}-\mu_\theta)
   +\tfrac1{\sigma^2}\lVert y-H\mu_{\theta\mid\mathcal D}\rVert^2
   +N\log\sigma^2\Bigr]
   -\tfrac{p}{2}\log 2\pi
   \]
   $$

2. **Generate five synthetic functions** \(f_1,\dots,f_5\) on 500 evenly-spaced points in \([-3,3]\) and corrupt them with Gaussian noise \(\eta\sim\mathcal N(0,\,\sigma^2),\ \sigma^2=0.25\).

3. **Evaluate the log-evidence** for polynomial bases of degree \(d=2,\dots,10\) (with a zero-mean, isotropic Gaussian prior \(\mathcal N(0,\alpha I)\), \(\alpha=1\)) for each function.

4. **Plot evidence vs. polynomial degree** per function and highlight the **best** and **worst** degree according to evidence.

5. **Fit BLR models** using the best and worst degrees, then plot:
   * the mean MMSE prediction,
   * a ±1 σ predictive standard-deviation band,
   * the noisy data.

6. **Estimate unknown sample noise** for a real-world temperature time-series:  
   * load a saved prior (7th-order polynomial) from `temp_prior.npy`,
   * sweep 100 noise variances \(\sigma^2\in[0.05,2]\),
   * plot log-evidence vs. noise variance and report the variance with maximal evidence,
   * discuss whether this variance is the *true* measurement noise.

---

## Requirements

* Python ≥ 3.8  
* numpy  
* matplotlib  

---

## Quick start
```bash
python bayesian_evidence.py
```

---

## What the script does
The script will:

* Generate plots for each synthetic function, saved interactively (or shown on screen).
* Print the best/worst polynomial degree per function.
* Display the evidence-vs-noise curve for the temperature data and report the peak evidence noise variance.


