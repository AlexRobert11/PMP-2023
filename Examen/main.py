import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
from scipy import stats



#Subiectul 1

# a.

# Încarcă setul de date într-un DataFrame
df = pd.read_csv("BostonHousing.csv", names=["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat","medv"])

print(df)

# b.

# Definim variabilele independente și dependente
X = df[['rm', 'crim', 'indus']].values
Y = df['medv'].values

with pm.Model() as model:
    # Priors pentru coeficienții regresiei
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=3)
    # Modelul liniar
    mu = alpha + pm.math.dot(X, beta)
    # Likelihood
    sigma = pm.HalfNormal('sigma', sd=10)
    medv = pm.Normal('medv', mu=mu, sd=sigma, observed=Y)

# Sample din distribuția posterioara
with model:
    trace = pm.sample(2000, tune=1000)

pm.summary(trace).round(2)

# c.

# Afisare distributii posterioare pentru coeficienti
pm.plot_posterior(trace, var_names=['beta'], credible_interval=0.95)
plt.show()

# Consider ca variabila care influenteaza cel mai multe rezultatul este rm, deoarece aceasta nu are
# valori de 0 si intervalul HDI pentru acest parametru este destul de restrans

# d.

# Extrageri simulate din distributia predictiva
with model:
    post_pred = pm.sample_posterior_predictive(trace, samples=1000)

# Calculam intervalul de predictie de 50% HDI
predictions = post_pred['medv']
hdi_50 = stats.hdi(predictions, credible_interval=0.5)

print("Intervalul de predictie de 50% HDI pentru valoarea locuintelor:", hdi_50)


# Subiectul 2

# a.
def posterior_grid_geometric(grid_points=50, heads=6, tails=9, geom_param=0.5):
    """
    A grid implementation for the coin-flipping problem with geometric prior
    """
    grid = np.linspace(0, 1, grid_points)
    prior = stats.geom.pmf(np.arange(1, grid_points + 1), geom_param)
    prior /= prior.sum()
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

# (0 pentru stema, 1 pentru cap)
data = np.array([1] * 3 + [0] * 10)  # Prima aparitie la a 4 a aruncare
points = 10
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid_geometric(points, h, t) # Calcularea distributiei geometrice
plt.plot(grid, posterior, 'o-')
plt.title(f'Prima stema la aruncarea a {h + 1} a')
plt.yticks([])
plt.xlabel('θ')
plt.show()


# b.
max_index = np.argmax(posterior)
max_theta = grid[max_index]
print("Valoarea lui θ care maximizează probabilitatea a posteriori este:", max_theta)
