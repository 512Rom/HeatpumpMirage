import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Constants
years = np.arange(21)  # 0 to 20 years
n_simulations = 1000  # number of Monte Carlo simulations
upfront_hp = 0  # in euros
upfront_gb = 0  # in euros
cost_per_kwh_gas = 0.15  # in euros
cop_hp_values = [2, 4, 6, 8]  # different COP values for the Heat Pump
cop_gb = 0.95
interest = 0.04  # 4%

# Heating demand (assumed to be 10000 kWh/year)
heating_demand_mean = 40000

# Standard deviation for the parameters (assumed values)
std_dev_electricity = 0.04
std_dev_gas = 0.04
std_dev_demand = 2000

# Different pairs of electricity and gas prices
electricity_prices = [0.46, 0.31, 0.21, 0.24, 0.18, 0.12, 0.10, 0.07, 0.05]
gas_prices = [0.15, 0.15, 0.15, 0.07, 0.07, 0.07, 0.03, 0.03, 0.03]

# Initialize the total costs with the new prices
total_cost_hp = np.zeros((len(electricity_prices), len(cop_hp_values), n_simulations, len(years)))
total_cost_gb = np.zeros((len(electricity_prices), n_simulations, len(years)))
# Run the Monte Carlo simulations with the new prices
np.random.seed(0)  # for reproducibility
for j in range(len(electricity_prices)):
    cost_per_kwh_electricity = electricity_prices[j]
    cost_per_kwh_gas = gas_prices[j]
    total_cost_hp[j, :, :, 0] = upfront_hp
    total_cost_gb[j, :, 0] = upfront_gb
    for i in range(1, len(years)):
        # Draw random values for the parameters
        cost_per_kwh_electricity_i = np.random.normal(cost_per_kwh_electricity, std_dev_electricity, n_simulations)
        cost_per_kwh_gas_i = np.random.normal(cost_per_kwh_gas, std_dev_gas, n_simulations)
        heating_demand_i = np.random.normal(heating_demand_mean, std_dev_demand, n_simulations)

        # Energy cost per year (heating demand / COP * cost per kWh)
        for k in range(len(cop_hp_values)):
            cop_hp = cop_hp_values[k]
            energy_cost_per_year_hp = heating_demand_i / cop_hp * cost_per_kwh_electricity_i
            total_cost_hp[j, k, :, i] = total_cost_hp[j, k, :, i-1] + energy_cost_per_year_hp / ((1 + interest) ** i)

        energy_cost_per_year_gb = heating_demand_i / cop_gb * cost_per_kwh_gas_i
        total_cost_gb[j, :, i] = total_cost_gb[j, :, i-1] + energy_cost_per_year_gb / ((1 + interest) ** i)

# Calculate the mean and standard deviation of the total costs
mean_total_cost_hp = total_cost_hp.mean(axis=2)
mean_total_cost_gb = total_cost_gb.mean(axis=1)
std_total_cost_hp = total_cost_hp.std(axis=2)
std_total_cost_gb = total_cost_gb.std(axis=1)

# Create a DataFrame to store the results
results = []

for j in range(len(electricity_prices)):
    for k in range(len(cop_hp_values)):
        mean_difference = mean_total_cost_hp[j, k, -1] - mean_total_cost_gb[j, -1]
        present_value_mean_difference = mean_difference / ((1 + interest) ** years[-1])
        results.append({
            'Electricity Price': electricity_prices[j],
            'Gas Price': gas_prices[j],
            'COP': cop_hp_values[k],
            'Discounted Present Value Difference': round(-present_value_mean_difference, 2)
        })

df = pd.DataFrame(results)

# Display the DataFrame
print(df)

latex_table = df.to_latex(index=False)

print(latex_table)

