import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

numar_eșantioane = 10000

distributii_timp_procesare = [
    stats.gamma(4, scale=1/3),
    stats.gamma(4, scale=1/2),
    stats.gamma(5, scale=1/2),
    stats.gamma(5, scale=1/3)
]
distributie_latenta = stats.expon(scale=1/4)

probabilitate_servere = [0.25, 0.25, 0.30, 0.20]

# Generăm eșantioane pentru X și numărăm câte dintre acestea sunt mai mari de 3 milisecunde
contor = 0
valori_X = []

for _ in range(numar_eșantioane):
    server_ales = np.random.choice(len(distributii_timp_procesare), p=probabilitate_servere)
    timp_procesare = distributii_timp_procesare[server_ales].rvs()
    latența = distributie_latenta.rvs()
    X = timp_procesare + latența
    valori_X.append(X)
    if X > 3:
        contor += 1

probabilitate_X_mai_mare_de_3 = contor / numar_eșantioane

plt.hist(valori_X, bins=50, density=True, alpha=0.6, color='b', label='Valori X')
plt.title('Densitatea distribuției lui X')
plt.xlabel('X (milisecunde)')
plt.ylabel('Densitate')
plt.legend()
plt.grid(True)
plt.show()

print(f"Probabilitatea ca X să fie mai mare de 3 milisecunde: {probabilitate_X_mai_mare_de_3}")
