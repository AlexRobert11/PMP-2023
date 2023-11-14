import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ArviZ as az

df = pd.read_csv('auto-mpg.csv')

X = df['horsepower'].values
y = df['mpg'].values

X = np.vstack((np.ones_like(X), X)).T

with pm.Model() as model:
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)

    mu = beta[0] + beta[1] * X[:, 1]

    mpg = pm.Normal('mpg', mu=mu, sd=1, observed=y)

# Model fitting
with model:
    trace = pm.sample(2000, tune=1000)

beta_mean = trace['beta'].mean(axis=0)

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 1], y, alpha=0.7, label='Observed Data')
plt.plot(X[:, 1], beta_mean[0] + beta_mean[1] * X[:, 1], color='red', label='Regression Line')
plt.title('Bayesian Linear Regression')
plt.xlabel('Horsepower')
plt.ylabel('Miles per Gallon')
plt.legend()
plt.grid(True)
plt.show()

pm.traceplot(trace)
plt.show()