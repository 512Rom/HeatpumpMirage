import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def plot_gas_usage(file_path, x):
    # Read the Excel file and create a pandas DataFrame
    df = pd.read_excel(file_path, engine='openpyxl')

    # Convert the gas usage from original gas meter units to HUF
    df['GasUsage'] = (df['GasUsage'] / 1000) * 82.74

    # Filter out rows where GasUsage is zero
    df = df[df['GasUsage'] != 0]

    # Convert the 'Date' column to a datetime object
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    # Calculate the moving average with a window of x days
    window = int(x * len(df) / (df.index[-1] - df.index[0]).days)
    df['MA'] = df['GasUsage'].rolling(window=window).mean()

    # Find the steepest increase in the moving average curve for the x-day period
    df['MA_diff'] = df['MA'].diff(periods=window)
    max_increase_idx = df['MA_diff'].idxmax()
    start_increase_idx = max_increase_idx - pd.Timedelta(days=x)

    # Find the nearest existing index in the DataFrame
    nearest_existing_idx = df.index.get_loc(start_increase_idx, method='nearest')
    steepest_increase = df.iloc[[nearest_existing_idx]]

    # Calculate average gas usage before the steepest increase (baseline usage)
    baseline_avg = df.loc[:steepest_increase.index[0], 'GasUsage'].mean()

    # Calculate heating gas usage by subtracting the baseline average from the total gas usage after the steepest increase
    df.loc[steepest_increase.index[0]:, 'Heating'] = df.loc[steepest_increase.index[0]:, 'GasUsage'] - baseline_avg
    df.loc[:steepest_increase.index[0], 'Heating'] = 0

    # Ensure that the heating gas usage values are not negative
    df['Heating'] = np.maximum(0, df['Heating'])

    # Plot the data, the moving average curve, and mark the steepest increase with a vertical line
    fig, ax = plt.subplots()

    ax.plot(df.index, df['GasUsage'], label='Total GasUsage')
    ax.plot(df.index, df['Heating'], label='Heating GasUsage', color='purple')
    ax.plot(df.index, df['MA'], label='Moving Average', color='red')
    ax.axvline(steepest_increase.index[0], color='green', linestyle='--', label='Steepest Increase')

    ax.set_title('Gas Usage with Moving Average')
    ax.set_xlabel('Date')
    ax.set_ylabel('Gas Usage (HUF)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Read the CSV file
    file_name = "2023-04-17-15-36_influxdb_data_modified.csv"
    temp_df = pd.read_csv(file_name)

    # Convert "_time" column to datetime format
    temp_df['_time'] = pd.to_datetime(temp_df['_time'], format='%d/%m/%Y %H:%M')


        # Filter the DataFrame to only include specified "_measurement" values
    selected_measurements = ["LivingRoomMotionSensor_SensorTemperature", "OneCallAPIweatherandforecast_Current_Temperature"]
    temp_df = temp_df[temp_df['_measurement'].isin(selected_measurements)]

    # Group the data by "_measurement" and "_time"
    grouped = temp_df.groupby(['_measurement', '_time']).mean().reset_index()

    # Create a list of unique "_measurement" values (sensors)
    measurements = grouped['_measurement'].unique()

    # Create a second y-axis
    ax2 = ax.twinx()

    # Map the original names to the new names
    label_map = {
        "LivingRoomMotionSensor_SensorTemperature": "Internal Temperature",
        "OneCallAPIweatherandforecast_Current_Temperature": "External Temperature"
    }

    # Plot the time series for each sensor on the second y-axis
    for measurement in measurements:
        sensor_data = grouped[grouped['_measurement'] == measurement]
        ax2.plot(sensor_data['_time'], sensor_data['_value'], label=label_map[measurement], linestyle='--')

    # Set the y-axis label for the second y-axis
    ax2.set_ylabel('Temperature')

    # Combine legends from both axes and place the legend outside of the plot
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', bbox_to_anchor=(1, 0.5))

    plt.show()

file_path = "GasMeter.xlsx"
x = 7  # You can change the value of x to any desired number of days
plot_gas_usage(file_path, x)

def compare_gas_usage(file_path, temp_file):
    # Read the Excel file and create a pandas DataFrame
    df = pd.read_excel(file_path, engine='openpyxl')

    # Convert the gas usage from original gas meter units to HUF
    df['GasUsage'] = (df['GasUsage'] / 1000) * 82.74

    # Filter out rows where GasUsage is zero
    df = df[df['GasUsage'] != 0]

    # Convert the 'Date' column to a datetime object
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    # Read the temperature data
    temp_df = pd.read_csv(temp_file)

    # Convert "_time" column to datetime format
    temp_df['_time'] = pd.to_datetime(temp_df['_time'], format='%d/%m/%Y %H:%M')

    # Filter the DataFrame to only include specified "_measurement" values
    selected_measurements = ["LivingRoomMotionSensor_SensorTemperature", "OneCallAPIweatherandforecast_Current_Temperature"]
    temp_df = temp_df[temp_df['_measurement'].isin(selected_measurements)]

    # Group the data by "_measurement" and "_time"
    grouped = temp_df.groupby(['_measurement', '_time']).mean().reset_index()

    # Convert "_time" column to datetime index
    grouped.set_index('_time', inplace=True)

    # Specify the date ranges
    start_date_1 = pd.to_datetime('2022-11-01')
    end_date_1 = pd.to_datetime('2023-01-11')
    start_date_2 = end_date_1 + pd.Timedelta(days=1)
    end_date_2 = start_date_2 + (end_date_1 - start_date_1)

    # Create boolean masks for each period
    mask_1 = (df.index >= start_date_1) & (df.index <= end_date_1)
    mask_2 = (df.index >= start_date_2) & (df.index <= end_date_2)

    # Calculate the total gas usage for each period
    total_gas_usage_1 = df.loc[mask_1, 'GasUsage'].sum()
    total_gas_usage_2 = df.loc[mask_2, 'GasUsage'].sum()

    # Calculate the number of days in each period
    days_1 = (df.index[mask_1].max() - df.index[mask_1].min()).days + 1
    days_2 = (df.index[mask_2].max() - df.index[mask_2].min()).days + 1

    # Calculate the average gas usage per day for each period
    avg_gas_usage_1 = total_gas_usage_1 / days_1
    avg_gas_usage_2 = total_gas_usage_2 / days_2

    print(f'Total gas usage from {start_date_1} to {end_date_1}: {total_gas_usage_1} HUF')
    print(f'Average gas usage per day from {start_date_1} to {end_date_1}: {avg_gas_usage_1} HUF/day')
    print(f'Total gas usage from {start_date_2} to {end_date_2}: {total_gas_usage_2} HUF')
    print(f'Average gas usage per day from {start_date_2} to {end_date_2}: {avg_gas_usage_2} HUF/day')

    # Filter the temperature data for each period
    temp_df_1 = grouped[(grouped.index >= start_date_1) & (grouped.index <= end_date_1)]
    temp_df_2 = grouped[(grouped.index >= start_date_2) & (grouped.index <= end_date_2)]

    # Calculate the average temperatures for each period
    avg_temp_1_internal = temp_df_1[temp_df_1['_measurement'] == "LivingRoomMotionSensor_SensorTemperature"]['_value'].mean()
    avg_temp_2_internal = temp_df_2[temp_df_2['_measurement'] == "LivingRoomMotionSensor_SensorTemperature"]['_value'].mean()

    avg_temp_1_external = temp_df_1[temp_df_1['_measurement'] == "OneCallAPIweatherandforecast_Current_Temperature"]['_value'].mean()
    avg_temp_2_external = temp_df_2[temp_df_2['_measurement'] == "OneCallAPIweatherandforecast_Current_Temperature"]['_value'].mean()

    print(f'Average internal temperature from {start_date_1} to {end_date_1}: {avg_temp_1_internal} Celsius')
    print(f'Average internal temperature from {start_date_2} to {end_date_2}: {avg_temp_2_internal} Celsius')
    print(f'Average external temperature from {start_date_1} to {end_date_1}: {avg_temp_1_external} Celsius')
    print(f'Average external temperature from {start_date_2} to {end_date_2}: {avg_temp_2_external} Celsius')

compare_gas_usage(file_path, '2023-04-17-15-36_influxdb_data_modified.csv')

