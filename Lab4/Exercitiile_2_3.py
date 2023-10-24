import numpy as np

# Exercitiul 1

lambda_clienti = 20
media_plasare_plata = 2.0  # Media timpului de plasare și plată a comenzii
deviatie_standard_plasare_plata = 0.5  # Deviația standard a timpului de plasare
media_pregatire_comanda = 10  # Media timpului de pregătire

numar_clienti = np.random.poisson(lambda_clienti)
numar_clienti_dupa_alpha = numar_clienti

timp_plasare_plata = np.random.normal(media_plasare_plata, deviatie_standard_plasare_plata, numar_clienti)

timp_pregatire_comanda = np.random.exponential(media_pregatire_comanda, numar_clienti)

timp_asteptare = timp_plasare_plata + timp_pregatire_comanda

timp_total_asteptare = np.sum(timp_asteptare)

timp_mediu_asteptare = timp_total_asteptare / numar_clienti

# Afișăm rezultatele
print(f"Numărul total de clienți: {numar_clienti}")
print(f"Timpul total de așteptare mediu: {timp_mediu_asteptare} minute")
print("Timpul total: ", timp_total_asteptare)

# Exercitiul 2

timp_maxim_servire = 15
probabilitate_dorita = 0.95

while True:
    numar_clienti = np.random.poisson(lambda_clienti)
    timp_plasare_plata = np.random.normal(media_plasare_plata, deviatie_standard_plasare_plata, numar_clienti)
    timp_pregatire_comanda = np.random.exponential(media_pregatire_comanda, numar_clienti)

    timp_total_servire = timp_plasare_plata + timp_pregatire_comanda

    probabilitate_actuala = np.mean(timp_total_servire <= timp_maxim_servire)

    if probabilitate_actuala >= probabilitate_dorita:
        break

    media_pregatire_comanda -= 0.1

print(f"Valoarea maximă a αlpha pentru a servi toți clienții în mai puțin de {timp_maxim_servire} minute cu o probabilitate de {probabilitate_dorita * 100}% este: {media_pregatire_comanda:.2f} minute")

# Exercitiul 3

timp_plasare_plata = np.random.normal(media_plasare_plata, deviatie_standard_plasare_plata, numar_clienti_dupa_alpha)
timp_pregatire_comanda = np.random.exponential(media_pregatire_comanda, numar_clienti_dupa_alpha)
timp_asteptare = timp_plasare_plata + timp_pregatire_comanda
timp_total_asteptare = np.sum(timp_asteptare)
timp_mediu_asteptare = timp_total_asteptare / numar_clienti

print(f"Numărul total de clienți: {numar_clienti_dupa_alpha}")
print(f"Timpul total de așteptare mediu: {timp_mediu_asteptare} minute")
print("Timpul total: ", timp_total_asteptare)
