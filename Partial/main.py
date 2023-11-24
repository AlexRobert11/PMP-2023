import random
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon

#Nu merge sa rulez programul, dupa ce importez un pachet din cele de mai sus, nu
# stiu exact imi da erori legate de diferite fisiere ale pycharm-ului

#Subiectul 1


#1.

def arunca_moneda(probabilitate):
    return random.choices([0, 1], [1 - probabilitate, probabilitate])[0]
    # Aici returneaza aleatoriu rezultatul avand in vedere probabilitatea data

def simuleaza_joc(numar_runde, probabilitate_p0):
    steme_p0 = 0 # numar de cazuri cand pica stema pentru jucatorul p0
    steme_p1 = 0 # numarul de cazuri cand pica stema pentru p1

    moneda_start = 1/2

    for runda in range(numar_runde):
        if (arunca_moneda(moneda_start) < 0.5):
            randul = 0  # Randul lui P0
        else:
            randul = 1  # Randul lui P1
        if randul == 0:
            steme_p0 += arunca_moneda(probabilitate_p0)
            for _ in range(2):
                steme_p1 += arunca_moneda(1/2)  # Arunca de 1 + 1 ori
        elif randul == 1:
            for _ in range(2):
                steme_p0 += arunca_moneda(probabilitate_p0)  # Arunca de 1 + 1 ori
            steme_p1 += arunca_moneda(1/2)

    return steme_p0, steme_p1

numar_runde_total = 20000
probabilitate_p0 = 1/3  # Probabilitatea stemei lui P0

rezultat = simuleaza_joc(numar_runde_total, probabilitate_p0)

print(f"Numarul total de steme obtinute de p0: {rezultat[0]}")
print(f"Numarul total de steme obtinute de p1: {rezultat[1]}")

#2.

# Definirea modelului Bayesian
model = BayesianModel([('Jucator', 'Probabilitate_P0'),
                       ('Probabilitate_P0', 'Steme_P0'),
                       ('Probabilitate_P0', 'Runda'),
                       ('Jucator', 'Probabilitate_P1'),
                       ('Probabilitate_P1', 'Steme_P1'),
                       ('Probabilitate_P1', 'Runda')])

# Aruncarea unei monede pentru a determina cine începe
moneda_start = 1/2
incepe_p0 = arunca_moneda(moneda_start) < 0.5

# Setarea valorilor inițiale pentru variabilele jucatorilor
valori_initiale = {'Jucator': 0 if incepe_p0 else 1,
                   'Probabilitate_P0': 1/3,
                   'Probabilitate_P1': 1/2}

# Adăugarea nodurilor la model
model.add_nodes_from(['Jucator', 'Probabilitate_P0', 'Probabilitate_P1', 'Runda', 'Steme_P0', 'Steme_P1'])
model.add_edges_from([('Jucator', 'Probabilitate_P0'),
                       ('Probabilitate_P0', 'Steme_P0'),
                       ('Probabilitate_P0', 'Runda'),
                       ('Jucator', 'Probabilitate_P1'),
                       ('Probabilitate_P1', 'Steme_P1'),
                       ('Probabilitate_P1', 'Runda')])

# Definirea CPD pentru Jucator
cpd_Jucator = TabularCPD('Jucator', 2, [[1, 0]] if incepe_p0 else [[0, 1]])

# Definirea CPD pentru Probabilitate_P0
cpd_Probabilitate_P0 = TabularCPD('Probabilitate_P0', 2, [[1, 0]])

# Definirea CPD pentru Probabilitate_P1
cpd_Probabilitate_P1 = TabularCPD('Probabilitate_P1', 2, [[1, 0]])

# Definirea CPD pentru Runda
cpd_Runda = TabularCPD('Runda', 2, [[1, 0], [0, 1]])

# Definirea CPD pentru Steme_P0
cpd_Steme_P0 = TabularCPD('Steme_P0', 2, [[1/3, 2/3]])

# Definirea CPD pentru Steme_P1
cpd_Steme_P1 = TabularCPD('Steme_P1', 2, [[1/2, 1/2]])

# Adăugarea CPD-urilor la model
model.add_cpds(cpd_Jucator, cpd_Probabilitate_P0, cpd_Probabilitate_P1, cpd_Runda, cpd_Steme_P0, cpd_Steme_P1)

# Verificarea modelului
print(model.check_model())


#3.

def simuleaza_joc(numar_runde, probabilitate_p0):
    stema = 0

    moneda_start = 1/2 # Probabilitatea monedei care decide cine incepe
    # Ignoram a doua runda, deoarece nu ne intereseaza
    for runda in range(numar_runde):
        if (arunca_moneda(moneda_start) < 0.5):
            randul = 0  # Randul lui P0
        else:
            randul = 1  # Randul lui P1
        if randul == 0:
            stema += arunca_moneda(probabilitate_p0) # Arunca moneda P0 si vede daca e stema
        elif randul == 1:
            stema += arunca_moneda(1/2) # Arunca moneda P1 si vede daca e stema

    return stema

numar_runde_total = 20000 # Numarul de runde
probabilitate_p0 = 1/3  # Probabilitatea stemei lui P0

rezultat = simuleaza_joc(numar_runde_total, probabilitate_p0)
nr_cap = 20000 - rezultat # Din nr total de steme scad si obtin de cate ori a picat cap
print("Numărul total de steme obținute:", rezultat)
print("Numărul total cand am obtinut cap:", nr_cap)


#Subiectul 2

'''Sa presupunem ca esti adiministratorul unui centru de service. Un aspect important pentru satisfactia clientilor este
timpul mediu de asteptare la telefon, pe care dorim sa-l estimam. Presupunem astfel pentru acest lucru un model de inferenta Bayesiana astfel:
'''

#1.

# Distribuție a priori pentru miu
prior_miu_mean = 10
prior_miu_std = 2
prior_miu = norm(loc=prior_miu_mean, scale=prior_miu_std)

# Distribuție a priori pentru sigma
prior_sigma_rate = 0.1
prior_sigma = expon(scale=1/prior_sigma_rate)

# Generarea a 200 de mostre
num_samples = 200

# Amestecarea parametrilor a priori
miu_samples = prior_miu.rvs(size=num_samples)
sigma_samples = prior_sigma.rvs(size=num_samples)

# Generarea timpilor de așteptare folosind distribuția normală
waiting_times = np.random.normal(loc=miu_samples, scale=sigma_samples)

# Afișarea rezultatelor
plt.hist(waiting_times, bins=20, density=True, alpha=0.7, color='b', label='Timpurile de asteptare')
plt.title('Distributia timpului de asteptare')
plt.xlabel('Timp de asteptare')
plt.ylabel('Probabilitatea distributiei')
plt.legend()
plt.show()

#2.

# Generarea datelor
np.random.seed(42)
observed_data = np.random.normal(loc=15, scale=3, size=200)

# Modelul PyMC
with pm.Model() as waiting_time_model:
    # Alegem distribuții normale pentru a priori ale lui miu și sigma
    prior_miu = pm.Normal('a_priori_miu', mu=15, sd=5)
    prior_sigma = pm.HalfNormal('a_priori_sigma', sd=5)

    # Distribuția a priori pentru datele observate
    waiting_times = pm.Normal('timpi_de_asteptare', mu=prior_miu, sd=prior_sigma, observed=observed_data)

    # Estimarea modelului
    trace = pm.sample(2000, tune=1000, cores=1)

# Afișarea rezultatelor
pm.traceplot(trace)
plt.show()


#3.

# Afișarea distribuției a posteriori a parametrului sigma
pm.plot_posterior(trace['a_priori_sigma'], credible_interval=0.95)
plt.title('Distributia a posteriori a lui sigma')
plt.xlabel('Sigma')
plt.ylabel('Densitatea a posteriori')
plt.show()

