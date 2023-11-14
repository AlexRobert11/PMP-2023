import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import numpy as np

df = pd.read_csv('auto-mpg.csv')

X = df['horsepower'].values
y = df['mpg'].values

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)

    mu = alpha + beta * X

    mpg = pm.Normal('mpg', mu=mu, sd=sigma, observed=y)

    # Model fitting
    trace = pm.sample(2000, tune=1000)

alpha_m = trace['alpha'].mean()
beta_m = trace['beta'].mean()

plt.figure(figsize=(15, 10))
plt.scatter(df['horsepower'], df['mpg'], alpha=0.7, label='Observed Data')
plt.plot(df['horsepower'], alpha_m + beta_m * df['horsepower'], c='k', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')

az.plot_hdi(df['horsepower'], trace['mpg_pred'].T, hdi_prob=0.95, color='k', fill_kwargs={'alpha': 0.2})

draws = range(0, trace['alpha'].size, 50)
for i in draws:
    plt.plot(df['horsepower'], trace['alpha'][i] + trace['beta'][i] * df['horsepower'], c='gray', alpha=0.2)

plt.title('Bayesian Linear Regression with 95% HDI and Posterior Samples')
plt.xlabel('Horsepower')
plt.ylabel('Miles per Gallon')
plt.legend()
plt.grid(True)
plt.show()
