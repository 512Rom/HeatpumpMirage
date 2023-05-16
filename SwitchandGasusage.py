import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor


# Read CSV and Excel files
csv_file = '2023-04-17-15-36_influxdb_data_modified.csv'
excel_file = 'GasMeter.xlsx'

csv_df = pd.read_csv(csv_file)
excel_df = pd.read_excel(excel_file, engine='openpyxl')
csv_df]

### 2

# Modify '_measurement' column
csv_df.loc[(csv_df['_time'] > '2023-01-11') & (csv_df['_value'] >= 43), '_measurement'] = 'GroundFloorUnderfloorHeatingControl_Switch'

# Filter rows and replace values > 0 with 1
filtered_df = csv_df[(csv_df['_measurement'].isin(['GroundFloorUnderfloorHeatingControl_Switch', 'GroundFloorRadiatorControl_Switch'])) & (csv_df['_value'] > 0)]
filtered_df['_value'] = np.where(filtered_df['_value'] > 0, 1, filtered_df['_value'])

# Extract 'Date' and 'GasUsage' columns from the excel_df, filtering out rows with 'GasUsage' equal to 0
gas_usage_df = excel_df[excel_df['GasUsage'] > 0][['Date', 'GasUsage']]

### 3
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

### 4
# Resample the merged data to daily
daily_data = merged_df.resample('D', on='_time').agg({'_value': 'sum', 'GasUsage': 'sum'}).reset_index()

# Prepare the features for the model
daily_data['DayOfWeek'] = daily_data['_time'].dt.dayofweek
daily_data['Month'] = daily_data['_time'].dt.month
X = daily_data[['DayOfWeek', 'Month', '_value']]
y = daily_data['GasUsage']

# Train the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X, y)

# Predict gas usage
daily_data['GasUsage_Pred'] = rf_regressor.predict(X)


### 5 

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Switch', color=color)

radiator_switch = merged_df[merged_df['_measurement'] == 'GroundFloorRadiatorControl_Switch']
ax1.scatter(radiator_switch['_time'], radiator_switch['_value'], color=color, alpha=0.5, label="Radiator Switch")

underfloor_switch = merged_df[merged_df['_measurement'] == 'GroundFloorUnderfloorHeatingControl_Switch']
ax1.scatter(underfloor_switch['_time'], underfloor_switch['_value'], color='tab:green', alpha=0.5, label="Underfloor Heating Switch")

ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Gas Usage', color=color)
ax2.scatter(merged_df['Date'], merged_df['GasUsage'], color=color, label="Gas Usage")
ax2.tick_params(axis='y', labelcolor=color)

ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
ax3.spines['right'].set_position(('outward', 60))
color = 'tab:orange'
ax3.set_ylabel('Predicted Gas Usage', color=color)
ax3.plot(daily_data['_time'], daily_data['GasUsage_Pred'], color=color, label="Predicted Gas Usage")
ax3.tick_params(axis='y', labelcolor=color)
fig.autofmt_xdate(rotation=45)

fig.tight_layout()
plt.show()



