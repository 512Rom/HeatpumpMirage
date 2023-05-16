import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
years = np.arange(21) # 0 to 20 years
n_simulations = 1000  # number of Monte Carlo simulations
upfront_hp_values = [10000, 25000, 40000] # in euros
upfront_gb_values = [4000, 7000, 10000] # in euros
cost_per_kwh_electricity_values = [0.46, 0.24, 0.10] # in euros
cost_per_kwh_gas_values = [0.15, 0.07, 0.03] # in euros
cop_hp_values = [2, 4, 6, 8]
interest = 0.04 # 5%
heating_demand_mean = 10000

# Standard deviation for the parameters (assumed values)
std_dev_electricity = 0.04
std_dev_gas = 0.04
std_dev_demand = 2000

# Initialize the total costs
total_cost_hp = np.zeros((len(cost_per_kwh_electricity_values), len(cop_hp_values), len(upfront_hp_values), len(years)))
total_cost_gb = np.zeros((len(cost_per_kwh_gas_values), len(upfront_gb_values), len(years)))

# Run the Monte Carlo simulations
np.random.seed(0)  # for reproducibility
for i in range(1, len(years)):
    for j in range(len(cost_per_kwh_electricity_values)):
        cost_per_kwh_electricity_i = np.random.normal(cost_per_kwh_electricity_values[j], std_dev_electricity, n_simulations)
        cost_per_kwh_gas_i = np.random.normal(cost_per_kwh_gas_values[j], std_dev_gas, n_simulations)
        heating_demand_i = np.random.normal(heating_demand_mean, std_dev_demand, n_simulations)
        for k in range(len(cop_hp_values)):
            cop_hp = cop_hp_values[k]
            for l in range(len(upfront_hp_values)):
                upfront_hp = upfront_hp_values[l]
                total_cost_hp[j, k, l, i] = total_cost_hp[j, k, l, i-1] + ((heating_demand_i / cop_hp * cost_per_kwh_electricity_i) / ((1 + interest) ** i))
                total_cost_hp[j, k, l, 0] = upfront_hp
            for m in range(len(upfront_gb_values)):
                upfront_gb = upfront_gb_values[m]
                total_cost_gb[j, m, i] = total_cost_gb[j, m, i-1] + ((heating_demand_i / cop_hp * cost_per_kwh_gas_i) / ((1 + interest) ** i))
                total_cost_gb[j, m, 0] = upfront_gb

# Calculate the mean and standard deviation of the total costs
mean_total_cost_hp = total_cost_hp.mean(axis=-2)
std_total_cost_hp = total_cost_hp.std(axis=-2)
mean_total_cost_gb = total_cost_gb.mean(axis=-2)
std_total_cost_gb = total_cost_gb.std(axis=-2)

# Create an empty DataFrame
import pandas as pd

# Prepare to store the results in a DataFrame
# Prepare to store the results in a DataFrame
df = pd.DataFrame(columns=['Electricity Cost', 'Gas Cost', 'COP', 'HP Investment', 'GB Investment', 
                           'HP Mean Total Cost (€)', 'HP Standard Deviation (€)',
                           'GB Mean Total Cost (€)', 'GB Standard Deviation (€)'])

for i in range(len(cost_per_kwh_electricity_values)):
    for j in range(len(cop_hp_values)):
        for k in range(len(upfront_hp_values)):
            for l in range(len(upfront_gb_values)):
                df = df.append({'Electricity Cost': cost_per_kwh_electricity_values[i],
                                'Gas Cost': cost_per_kwh_gas_values[i],
                                'COP': cop_hp_values[j],
                                'HP Investment': upfront_hp_values[k],
                                'GB Investment': upfront_gb_values[l],
                                'HP Mean Total Cost (€)': mean_total_cost_hp[i, j, k, -1],
                                'HP Standard Deviation (€)': std_total_cost_hp[i, j, k, -1],
                                'GB Mean Total Cost (€)': mean_total_cost_gb[i, l, -1],
                                'GB Standard Deviation (€)': std_total_cost_gb[i, l, -1]},
                               ignore_index=True)

# Export the DataFrame to a LaTeX table
with open('table.tex', 'w') as tf:
    tf.write(df.to_latex(index=False))
