import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Parametrii pentru distribuția exponentială
lambda1 = 4
lambda2 = 6

probabilitate_primul_mecanic = 0.4

numar_clienti = 10000

valori_X = []
for i in range(numar_clienti):
    if np.random.rand() < probabilitate_primul_mecanic:
        valori_X.append(stats.expon(scale=1/lambda1).rvs())
    else:
        valori_X.append(stats.expon(scale=1/lambda2).rvs())

media_X = np.mean(valori_X)
deviatia_standard_X = np.std(valori_X)

# Realizăm un grafic al densității distribuției lui X
plt.hist(valori_X, bins=30, density=True, alpha=0.8, color='g', label='Valori X')
plt.title('Densitatea distribuției lui X')
plt.xlabel('X')
plt.ylabel('Densitate')
plt.legend()
plt.grid(True)
plt.show()

print(f"Media lui X: {media_X}")
print(f"Deviatia standard a lui X: {deviatia_standard_X}")
