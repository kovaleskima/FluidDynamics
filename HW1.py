import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#First, import and read csv data 
data = pd.read_csv('lab1-water-spout-measurements.csv')
data.reset_index(drop=True, inplace=True)
data = data.dropna(subset=["height (cm)"])
print(data)

#Fit a quadratic curve to the data
coefficients = np.polyfit(data['seconds'], data['height (cm)'], 2)
quadratic_function = np.poly1d(coefficients)

#Extract leading coefficient so that we can calculate our radius
leading_coefficient = coefficients[0]
print("Leading coefficient a = ", leading_coefficient)

#Create a new column in data containing the fitted values
data['y_fit'] = quadratic_function(data['seconds'])

#Plot the original data and the fitted curve
plt.scatter(data['seconds'], data['height (cm)'], label='Observed h(t)')
plt.plot(data['seconds'], data['y_fit'], label='Predicted h(t)', color='red')
plt.xlabel("time in seconds")
plt.ylabel("height in cm")
plt.legend()
plt.show()

print(data)