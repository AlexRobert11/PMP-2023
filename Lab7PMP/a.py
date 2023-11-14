import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

df = pd.read_csv('auto-mpg.csv')

print(df.head())

print(df.info())

plt.figure(figsize=(15, 10))
plt.scatter(df['horsepower'], df['mpg'], alpha=0.7)
plt.title('Relația dintre cp si mpg')
plt.xlabel('cp')
plt.ylabel('mpg')
plt.grid(True)
plt.show()




