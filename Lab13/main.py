import arviz as az
import matplotlib.pyplot as plt

centered_data = az.load_arviz_data("centered_eight")
non_centered_data = az.load_arviz_data("non_centered_eight")

# 1.
num_chains_centered = centered_data.posterior.chain.size
total_samples_centered = centered_data.posterior.draw.size
num_chains_non_centered = non_centered_data.posterior.chain.size
total_samples_non_centered = non_centered_data.posterior.draw.size

print(f"Model Centrat: Număr de lanțuri: {num_chains_centered}, Mărime totală a eșantionului: {total_samples_centered}")
print(f"Model Necentrat: Număr de lanțuri: {num_chains_non_centered}, Mărime totală a eșantionului: {total_samples_non_centered}")

# Distribuția a posteriori
az.plot_posterior(centered_data, var_names=['mu', 'tau'], figsize=(10, 6), textsize=14, round_to=2, hdi_prob=0.95, rope=[-0.1, 0.1])
az.plot_posterior(non_centered_data, var_names=['mu', 'tau'], figsize=(10, 6), textsize=14, round_to=2, hdi_prob=0.95, rope=[-0.1, 0.1])
plt.show()

# 2. Comparați modelele folosind Rhat și autocorelația pentru parametrii mu și tau
az.summary(centered_data, var_names=['mu', 'tau'], round_to=2, hdi_prob=0.95).head()

az.summary(non_centered_data, var_names=['mu', 'tau'], round_to=2, hdi_prob=0.95).head()

# 3.
divergences_centered = centered_data.sample_stats.diverging.sum()
divergences_non_centered = non_centered_data.sample_stats.diverging.sum()

print(f"Numărul de divergențe pentru modelul centrat: {divergences_centered}")
print(f"Numărul de divergențe pentru modelul necentrat: {divergences_non_centered}")

# Vizualizarea divergențelor în spațiul parametrilor folosind plot_pair
az.plot_pair(centered_data, var_names=['mu', 'tau'], divergences=True)
az.plot_pair(non_centered_data, var_names=['mu', 'tau'], divergences=True)
plt.show()
