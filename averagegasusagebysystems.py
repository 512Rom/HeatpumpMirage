import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Read CSV and Excel files
csv_file = '2023-04-17-15-36_influxdb_data_modified.csv'
excel_file = 'GasMeter.xlsx'

csv_df = pd.read_csv(csv_file)
excel_df = pd.read_excel(excel_file, engine='openpyxl')

# Modify '_measurement' column
csv_df.loc[(csv_df['_time'] > '2023-01-11') & (csv_df['_value'] >= 43), '_measurement'] = 'GroundFloorUnderfloorHeatingControl_Switch'

# Filter rows and replace values > 0 with 1
filtered_df = csv_df[(csv_df['_measurement'].isin(['GroundFloorUnderfloorHeatingControl_Switch', 'GroundFloorRadiatorControl_Switch'])) & (csv_df['_value'] > 0)]
filtered_df['_value'] = np.where(filtered_df['_value'] > 0, 1, filtered_df['_value'])

# Extract 'Date' and 'GasUsage' columns from the excel_df, filtering out rows with 'GasUsage' equal to 0
gas_usage_df = excel_df[excel_df['GasUsage'] > 0][['Date', 'GasUsage']]

# Merge filtered_df and gas_usage_df on date
filtered_df['_time'] = pd.to_datetime(filtered_df['_time'])
gas_usage_df['Date'] = pd.to_datetime(gas_usage_df['Date'])

merged_df = pd.merge(filtered_df, gas_usage_df, left_on='_time', right_on='Date', how='outer')
merged_df.sort_values(by='_time', inplace=True)

# Get the first and last date of the GasUsage
start_date = gas_usage_df['Date'].min()
end_date = gas_usage_df['Date'].max()

# Filter the merged dataframe based on start_date and end_date
merged_df = merged_df[(merged_df['_time'] >= start_date) & (merged_df['_time'] <= end_date)]

# Calculate the average gas usage when the underfloor heating switch is on versus off
underfloor_on_mean = merged_df[merged_df['_measurement'] == 'GroundFloorUnderfloorHeatingControl_Switch'].groupby('_value')['GasUsage'].mean()
underfloor_off_mean = merged_df[merged_df['_measurement'] != 'GroundFloorUnderfloorHeatingControl_Switch'].groupby('_value')['GasUsage'].mean()

# Calculate the average gas usage when the radiator switch is on versus off
radiator_on_mean = merged_df[merged_df['_measurement'] == 'GroundFloorRadiatorControl_Switch'].groupby('_value')['GasUsage'].mean()
radiator_off_mean = merged_df[merged_df['_measurement'] != 'GroundFloorRadiatorControl_Switch'].groupby('_value')['GasUsage'].mean()

# Plot the average gas usage when the underfloor heating switch is on versus off
fig, ax = plt.subplots()
ax.bar(['Underfloor Heating On', 'Underfloor Heating Off'], [underfloor_on_mean[1], underfloor_off_mean[1]], color='tab:green')
ax.bar(['Radiator On', 'Radiator Off'], [radiator_on_mean[1], radiator_off_mean[1]], color='tab:red')
ax.set_ylabel('Average Gas Usage')
ax.set_title('Average Gas Usage with Underfloor Heating and Radiator On vs Off')
plt.show()
