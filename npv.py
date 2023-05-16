import matplotlib.pyplot as plt
import numpy as np

# Constants
years = np.arange(21) # 0 to 20 years
upfront_hp = 40000 # in euros
upfront_gb = 5000 # in euros
cost_per_kwh_electricity = 0.095 # in euros
cost_per_kwh_gas = 0.0243 # in euros
cop_hp = 3
cop_gb = 0.95
interest = 0.05 # 5%

# Heating demand (assumed to be x kWh/year)
heating_demand = 5000

# Energy cost per year (heating demand / COP * cost per kWh)
energy_cost_per_year_hp = heating_demand / cop_hp * cost_per_kwh_electricity
energy_cost_per_year_gb = heating_demand / cop_gb * cost_per_kwh_gas

# Total cost over time (upfront + cumulative energy cost)
total_cost_hp = np.zeros(21)
total_cost_gb = np.zeros(21)

total_cost_hp[0] = upfront_hp
total_cost_gb[0] = upfront_gb

for i in range(1, 21):
    total_cost_hp[i] = total_cost_hp[i-1] + energy_cost_per_year_hp / ((1 + interest) ** i)
    total_cost_gb[i] = total_cost_gb[i-1] + energy_cost_per_year_gb / ((1 + interest) ** i)

# Plot
plt.figure(figsize=(10,6))
plt.plot(years, total_cost_hp, label='Heat Pump')
plt.plot(years, total_cost_gb, label='Gas Boiler')
plt.xlabel('Years')
plt.ylabel('Total Cost (€)')
plt.title('Total Cost of Heat Pump vs. Gas Boiler over Time')
plt.legend()
plt.grid(True)
plt.show()
