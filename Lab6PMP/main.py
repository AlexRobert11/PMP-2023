import pymc3 as pm
import arviz as az

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

with pm.Model() as model:
    n = pm.Poisson('n', mu=10)

    for Y in Y_values:
        for theta in theta_values:
            Y_observed = pm.Binomial(f'Y_observed_Y{Y}_Theta{theta}', n=n, p=theta, observed=Y)

# MCMC sampling
with model:
    trace = pm.sample(1000, tune=1000, cores=1)  # Aici putem ajusta numărul de iterații și burn-in

# Vizualizare distribuția a posteriori
az.plot_posterior(trace)