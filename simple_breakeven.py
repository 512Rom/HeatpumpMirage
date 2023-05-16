import numpy as np
import matplotlib.pyplot as plt

# Constants
gas_cost_per_kwh = 0.10
electricity_cost_per_kwh = 0.11
heat_pump_install_cost = 15000
gas_boiler_install_cost = 3000

# Assumed efficiency values (You can replace these with actual values if available)
AFUE = 0.9
COP = 3.5

# Function to calculate the total cost for each system
def total_cost(heating_demand, years, initial_cost, cost_per_kwh, efficiency):
    annual_energy_cost = heating_demand * cost_per_kwh / efficiency
    return initial_cost + (years * annual_energy_cost)

# Calculate break-even point
years = np.arange(1, 30)
gas_costs = total_cost(heating_demand=15000, years=years, initial_cost=gas_boiler_install_cost,
                       cost_per_kwh=gas_cost_per_kwh, efficiency=AFUE)
heat_pump_costs = total_cost(heating_demand=15000, years=years, initial_cost=heat_pump_install_cost,
                             cost_per_kwh=electricity_cost_per_kwh, efficiency=COP)

break_even_years = np.where(np.diff(np.sign(gas_costs - heat_pump_costs)))[0][0] + 1

# Plot the total costs and break-even point
plt.plot(years, gas_costs, label="Gas Boiler")
plt.plot(years, heat_pump_costs, label="Heat Pump")
plt.axvline(x=break_even_years, color="red", linestyle="--", label=f"Break-even point: {break_even_years} years")

plt.xlabel("Years")
plt.ylabel("Total Cost (€)")
plt.title("Break-even Analysis: Gas Boiler vs. Air-to-Water Heat Pump")
plt.legend()
plt.grid()
plt.show()
