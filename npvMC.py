import matplotlib.pyplot as plt
import numpy as np

# Constants
years = np.arange(21) # 0 to 20 years
n_simulations = 1000  # number of Monte Carlo simulations
upfront_hp = 0 # in euros
upfront_gb = 0 # in euros
cost_per_kwh_electricity = 0.42 # in euros
cost_per_kwh_gas = 0.15 # in euros
cop_hp = 6
cop_gb = 0.95
interest = 0.04 # 5%

# Heating demand (assumed to be 20000 kWh/year)
heating_demand_mean = 20000

# Standard deviation for the parameters (assumed values)
std_dev_electricity = 0.04
std_dev_gas = 0.04
std_dev_demand = 2000

# Initialize the total costs
total_cost_hp = np.zeros((n_simulations, len(years)))
total_cost_gb = np.zeros((n_simulations, len(years)))

total_cost_hp[:, 0] = upfront_hp
total_cost_gb[:, 0] = upfront_gb

# Run the Monte Carlo simulations
np.random.seed(0)  # for reproducibility
for i in range(1, len(years)):
    # Draw random values for the parameters
    cost_per_kwh_electricity_i = np.random.normal(cost_per_kwh_electricity, std_dev_electricity, n_simulations)
    cost_per_kwh_gas_i = np.random.normal(cost_per_kwh_gas, std_dev_gas, n_simulations)
    heating_demand_i = np.random.normal(heating_demand_mean, std_dev_demand, n_simulations)

    # Energy cost per year (heating demand / COP * cost per kWh)
    energy_cost_per_year_hp = heating_demand_i / cop_hp * cost_per_kwh_electricity_i
    energy_cost_per_year_gb = heating_demand_i / cop_gb * cost_per_kwh_gas_i

    total_cost_hp[:, i] = total_cost_hp[:, i-1] + energy_cost_per_year_hp / ((1 + interest) ** i)
    total_cost_gb[:, i] = total_cost_gb[:, i-1] + energy_cost_per_year_gb / ((1 + interest) ** i)

# Calculate the mean and standard deviation of the total costs
mean_total_cost_hp = total_cost_hp.mean(axis=0)
std_total_cost_hp = total_cost_hp.std(axis=0)
mean_total_cost_gb = total_cost_gb.mean(axis=0)
std_total_cost_gb = total_cost_gb.std(axis=0)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(years, mean_total_cost_hp, label='Heat Pump', color='blue')
plt.fill_between(years, mean_total_cost_hp - std_total_cost_hp, mean_total_cost_hp + std_total_cost_hp, color='blue', alpha=0.2)
plt.plot(years, mean_total_cost_gb, label='Gas Boiler', color='orange')
plt.fill_between(years, mean_total_cost_gb - std_total_cost_gb, mean_total_cost_gb + std_total_cost_gb, color='orange', alpha=0.2)
plt.xlabel('Years')
plt.ylabel('Total Cost (€)')
plt.title('Total Cost of Heat Pump vs. Gas Boiler over Time (Monte Carlo Simulation)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the difference in total costs at the end of the 20 year period
total_cost_difference = abs(total_cost_hp[:, -1] - total_cost_gb[:, -1])

# Calculate the mean and standard deviation of the difference
mean_difference = total_cost_difference.mean()
std_difference = total_cost_difference.std()

print("Mean difference in total cost (HP - GB) after 20 years: {:.2f} €".format(mean_difference))
print("Standard deviation of the difference in total cost after 20 years: {:.2f} €".format(std_difference))

# Calculate the present value of the mean difference
present_value_mean_difference = mean_difference / ((1 + interest) ** years[-1])

print("Present value of the mean difference in total cost (HP - GB): {:.2f} €".format(present_value_mean_difference))
