import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup Table and do some conversions to si
df = pd.read_csv("./lab3data.csv")
df['x'] = np.abs(df['x']) #fix negative x's because trackers coordinate system is wack
print(df)
