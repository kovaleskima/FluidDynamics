import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

diameter = 0.012 #pipe diameter is 12mm
radius = 0.006 #pipe radius is 6mm
mu = 1e-6 #viscous constant
L = 3.02 #length of the pipe was 3.02m

#create function for velocity profile u taking r and dp as inputs
def velocity_profile(r, dp):
    return -dp*(r**2)/(4*mu*L) + (dp*(radius**2))/(4*mu*L)

#create function for volume transport Q taking dp as an input
def idealized_Q(r, dp):
    return np.pi*dp*(r**4)/(8*mu*L)

# Setup Table and do some conversions to si
df = pd.read_csv("./lab2-data.csv")
df.insert(3, 'Q (L/s)', df['V (L)']/df['T (s)'])

df.insert(4, 'Reynolds', 0) #insert a new column
df['Reynolds'] = df['Q (L/s)'] / (np.pi * (radius**2)) #calculate Reynold's number
df['Reynolds'] = ((df['Reynolds']*diameter)/mu)*0.001 #divide by 1000 to convert to cubic meters/s

print(df)

for index, row in df.iterrows():
    Q = row['Q (L/s)']  # Extract the value from the column
    dp = row['Pressure difference (m)']
    # Define r values for the plot
    r = np.linspace(-radius, radius, 200)
    u = velocity_profile(r, dp)
    # Plot the function, keeping all functions on the same graph
    #plt.plot(r, u, label=f'Q='+str(Q))

# Add labels, title, and legend outside the loop
#plt.xlabel('Radius (m)')
#plt.ylabel('Velocity Profile u (m/s)')
#plt.title('Velocity Profiles w.r.t. Measured Flow Rate')
#plt.legend(loc='upper right')  # Show the legend for all functions
#plt.grid(True)
#plt.show()

# Calculate ideal Q based on Poiseuille's law
# Note: 1m of pressure difference equates to 9804.13943Pa
df.insert(5, 'Ideal Q', idealized_Q(radius, df['Pressure difference (m)']*9804.13943)) 
print(df)
plt.scatter(df['Pressure difference (m)'], df['Q (L/s)'], label=f'Measured Q')
plt.scatter(df['Pressure difference (m)'], df['Ideal Q'], label=f'Ideal Q')

plt.xlabel('Pressure Difference (Pa)')
plt.ylabel('Flow Rate Q (L/s)')
plt.title('Flow Rate w.r.t. Pressure Difference')
plt.legend(loc='upper left')  # Show the legend for all functions
plt.grid(True)
plt.show()