import numpy as np
import scipy.stats as stats

lambda_clienti = 20
media_plasare_plata = 2.0  # Media timpului de plasare și plată a comenzii
deviatie_standard_plasare_plata = 0.5  # Deviația standard a timpului de plasare
media_pregatire_comanda = 10  # Media timpului de pregătire

numar_clienti = np.random.poisson(lambda_clienti)

timp_plasare_plata = np.random.normal(media_plasare_plata, deviatie_standard_plasare_plata, numar_clienti)

timp_pregatire_comanda = np.random.exponential(media_pregatire_comanda, numar_clienti)

timp_asteptare = timp_plasare_plata + timp_pregatire_comanda

timp_total_asteptare = np.sum(timp_asteptare)

timp_mediu_asteptare = timp_total_asteptare / numar_clienti

# Afișăm rezultatele
print(f"Numărul total de clienți: {numar_clienti}")
print(f"Timpul total de așteptare mediu: {timp_mediu_asteptare} minute")
print("Timpul total: ", timp_total_asteptare)
