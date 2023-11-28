import pandas as pd
import pymc3 as pm
import numpy as np

df = pd.read_csv('Prices.csv')

print(df.head())

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10)
    beta2 = pm.Normal('beta2', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)

    mu = alpha + beta1 * df['Processor_Frequency'] + beta2 * np.log(df['Hard_Disk_Size'])

    prices = pm.Normal('prices', mu=mu, sd=sigma, observed=df['Sale_Price'])

with model:
    trace = pm.sample(2000, tune=1000)


beta1_hdi = pm.stats.hpd(trace['beta1'])
beta2_hdi = pm.stats.hpd(trace['beta2'])

print("Interval de credibilitate 95% pentru beta1:", beta1_hdi)
print("Interval de credibilitate 95% pentru beta2:", beta2_hdi)