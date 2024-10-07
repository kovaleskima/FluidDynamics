import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#First, import and read csv data 
data = pd.read_csv('lab1-water-spout-measurements.csv')
data = data.dropna(subset=["height (cm)"])
data.reset_index(drop=True)

# Fit a quadratic curve to the data
coefficients = np.polyfit(data['seconds'], data['height (cm)'], 2)
quadratic_function = np.poly1d(coefficients)

# Create a new column in the DataFrame with the fitted values
data['y_fit'] = np.poly1d(data['seconds'].values, data['height (cm)'].values)

# Plot the original data and the fitted curve
plt.scatter(data['seconds'], data['height (cm)'], label='Lab Data')
plt.plot(data['seconds'], data['y_fit'], label='Quadratic Fit', color='red')
plt.legend()
plt.show()

print(data)