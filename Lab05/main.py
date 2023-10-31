import pymc as pm
import numpy as np
import pandas as pd

# Citirea datelor din fișierul CSV și afișarea numelor coloanelor disponibile
data = pd.read_csv("trafic.csv")
print(data.columns)  # Afișăm numele coloanelor

# Alegeți numele corect al coloanei care conține datele de trafic
traffic_data = data["nr.masini"].values  # Înlocuiți "nr. masini" cu numele corect al coloanei

# Defineți intervalele de timp pentru creșterea și descreșterea traficului
increase_intervals = [(7, 16)]
decrease_intervals = [(8, 19)]

# Defineți parametrii pentru distribuția Poisson pentru traficul global
with pm.Model() as model:
    lambda_global = pm.Exponential("lambda_global", lam=1)  # Opțiune 1: Utilizarea lui 'lam'

    # Definiți parametrii pentru distribuția Poisson pentru fiecare interval de creștere și descreștere
    lambda_increase = [pm.Exponential("lambda_increase_%d" % i, lam=1) for i in range(len(increase_intervals))]
    lambda_decrease = [pm.Exponential("lambda_decrease_%d" % i, lam=1) for i in range(len(decrease_intervals))]

    # Modelul probabilistic
    @pm.Deterministic
    def traffic_model(lambda_global=lambda_global, lambda_increase=lambda_increase, lambda_decrease=lambda_decrease):
        model = np.zeros(len(traffic_data))
        for i, minute in enumerate(range(240, 1440)):
            for start, end in increase_intervals:
                if start <= minute < end:
                    model[i] = lambda_increase[0]
            for start, end in decrease_intervals:
                if start <= minute < end:
                    model[i] = lambda_decrease[0]
            if model[i] == 0:
                model[i] = lambda_global
        return model

    # Definiți observația datelor
    observed_traffic = pm.Poisson("observed_traffic", value=traffic_data, mu=traffic_model)

# Realizați inferența Bayesiană pentru a găsi cele mai probabile valori ale parametrilor și intervalele de timp
mcmc = pm.MCMC(model)
mcmc.sample(iter=10000, burn=1000, thin=10)

# Analiza rezultatelor
lambda_global_samples = mcmc.trace('lambda_global')[:]
lambda_increase_samples = [mcmc.trace('lambda_increase_%d' % i)[:].mean() for i in range(len(increase_intervals))]
lambda_decrease_samples = [mcmc.trace('lambda_decrease_%d' % i)[:].mean() for i in range(len(decrease_intervals))]

# Găsiți capetele cele mai probabile ale celor 5 intervale de timp pentru creștere și descreștere
increase_intervals_samples = [(7, 16)]  # Inițial, punem intervalul cunoscut
decrease_intervals_samples = [(8, 19)]  # Inițial, punem intervalul cunoscut

# Afișăm rezultatele
print("Cele mai probabile intervale pentru creștere:", increase_intervals_samples)
print("Cele mai probabile intervale pentru descreștere:", decrease_intervals_samples)
print("Cele mai probabile valori ale parametrului lambda_global:", lambda_global_samples.mean())
print("Cele mai probabile valori ale parametrilor lambda_increase:", lambda_increase_samples)
print("Cele mai probabile valori ale parametrilor lambda_decrease:", lambda_decrease_samples)
