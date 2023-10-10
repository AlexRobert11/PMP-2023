import numpy as np
import matplotlib.pyplot as plt

def arunca_moneda(p):
    return 's' if np.random.rand() < p else 'b'

numar_experimente = 500

numar_aruncari = 10

rezultate = []

for _ in range(numar_experimente):
    rezultat_experiment = ''
    for _ in range(numar_aruncari):
        rezultat_experiment += arunca_moneda(0.5)
        rezultat_experiment += arunca_moneda(0.3)
        #rezultat_experiment += " "
    rezultate.append(rezultat_experiment)

distributie = {'ss': 0, 'sb': 0, 'bs': 0, 'bb': 0}
for rezultat in rezultate:
    distributie['ss'] += rezultat.count('ss')
    distributie['sb'] += rezultat.count('sb')
    distributie['bs'] += rezultat.count('bs')
    distributie['bb'] += rezultat.count('bb')

plt.bar(distributie.keys(), distributie.values())
plt.xlabel('Rezultat')
plt.ylabel('Frecvență')
plt.title('Distribuția variabilelor aleatoare')
plt.show()
