import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

data = np.loadtxt('data.csv')

# 1

order = 5
x = data
y = np.ones_like(x)

x_p = np.vstack([x**i for i in range(1, order+1)])
x_s = (x_p - x_p.mean(axis=1, keepdims=True)) / x_p.std(axis=1, keepdims=True)
y_s = (y - y.mean()) / y.std()

plt.scatter(x_s[0], y_s)
plt.xlabel('x')
plt.ylabel('y')

# 2a
with pm.Model() as model_p:
    beta = pm.Normal('beta', mu=0, sd=100, shape=order)
    mu = pm.Deterministic('mu', pm.math.dot(beta, x_p))
    sigma = pm.HalfCauchy('sigma', 5)
    obs = pm.Normal('obs', mu=mu, sd=sigma, observed=y_s)

    trace_p_sd_100 = pm.sample(1000, tune=1000)

# 2b
with pm.Model() as model_p_sd_array:
    beta = pm.Normal('beta', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
    mu = pm.Deterministic('mu', pm.math.dot(beta, x_p))
    sigma = pm.HalfCauchy('sigma', 5)
    obs = pm.Normal('obs', mu=mu, sd=sigma, observed=y_s)
    trace_p_sd_array = pm.sample(1000, tune=1000)

# 3
data_500 = np.loadtxt('./data/data_500.csv')
order_500 = 5
x_500 = data_500
y_500 = np.ones_like(x_500)
x_p_500 = np.vstack([x_500**i for i in range(1, order_500+1)])
x_s_500 = (x_p_500 - x_p_500.mean(axis=1, keepdims=True)) / x_p_500.std(axis=1, keepdims=True)
y_s_500 = (y_500 - y_500.mean()) / y_500.std()
plt.figure()
plt.scatter(x_s_500[0], y_s_500)
plt.xlabel('x')
plt.ylabel('y')