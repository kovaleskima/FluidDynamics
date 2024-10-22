import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import x & y and convert from cm to m
df = pd.read_csv("./lab3data.csv")
df['x'] = 0.01*np.abs(df['x']) #fix negative x's because trackers coordinate system is wack
df['y'] = 0.01*np.abs(df['y'])

#given constants:
g = 9.8 #m/s^2
rho = 997 #kg/m^3

# flow rate Q was recorded to be 460mL/2.7s, convert to cubic meters/s
Q = (460/2.7)*(0.001**2)
# width of the pipe recorded to be 5.5cm, convert to m
width = 5.5*0.01
# flow rate / cross sectional area = avg velocity
df.insert(3, 'U(x)', 0)
df.insert(4, 'Fr(x)', 0)
df.insert(5, 'E(x)', 0)

df['U(x)'] = Q/(df['y']*width) #m/s
df['Fr(x)'] = df['U(x)']/(np.sqrt(g*df['y']))
df['E(x)'] = rho*g*df['y'] + 0.5*rho*(df['U(x)']**2) #Joules per m^3

#fig, ax1 = plt.subplots()

# Plot on the first x-axis
#ax1.plot(df['x'], df['Fr(x)'], 'g-', label='Fr(x)')
#ax1.set_ylabel('Froude Number', color='g')
#ax1.set_xlabel('x (m)')

# Create a twin axis that shares the same y-axis
#ax2 = ax1.twinx()

# Plot on the second x-axis
#ax2.plot(df['x'], df['E(x)'], 'b-', label='E(x)')
#ax2.set_ylabel('Energy (J/m^3)', color='b')

#plt.title('Froude Number & Energy w.r.t. x')
#plt.show()

# import Q, H1, H2 and convert to SI units
df = pd.read_csv("./lab3heights.csv")
df['Q'] = (1e-6)*df['Q']
df['H1'] = 0.001*df['H1']
df['H2'] = 0.001*df['H2']

#time to do some math
df.insert(3, 'H2/H1', 0)
df['H2/H1'] = df['H2']/df['H1']

df.insert(4, 'Fr^2', 0)
df['Fr^2'] = (df['Q']**2)/(g*(df['H1']**3))

df.insert(5, 'Predicted H2/H1', 0)
df['Predicted H2/H1'] = 0.5*(-1+np.sqrt(1+8*df['Fr^2']))
print(df)