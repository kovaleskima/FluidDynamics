import numpy as np
import pandas as pd

# Setup Table and do some conversions to si
df = pd.read_csv("./lab2-data.csv")
df.insert(3, 'Q (L/s)', df['V (L)']/df['T (s)'])
print(df)