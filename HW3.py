import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import x & y and convert from cm to m
df = pd.read_csv("./lab3data.csv")
df['x'] = 0.01*np.abs(df['x']) #fix negative x's because trackers coordinate system is wack
df['y'] = 0.01*np.abs(df['y'])
print(df) #sanity check


