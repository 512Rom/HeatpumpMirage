import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
def plot_gas_usage(file_path, x):
    # Read the Excel file and create a pandas DataFrame
    df = pd.read_excel(file_path, engine='openpyxl')

    # Filter out rows where GasUsage is zero
    df = df[df['GasUsage'] != 0]

    # Convert the 'Date' column to a datetime object
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    # Calculate the moving average with a window of x days
    # window = int(x * len(df) / (df.index[-1] - df.index[0]).days)
    # df['MA'] = df['GasUsage'].rolling(window=window).mean()

    # Find the steepest increase in the moving average curve for the x-day period
    # df['MA_diff'] = df['MA'].diff(periods=window)
    # max_increase_idx = df['MA_diff'].idxmax()
    # start_increase_idx = max_increase_idx - pd.Timedelta(days=x)

    # Find the nearest existing index in the DataFrame
    # nearest_existing_idx = df.index.get_loc(start_increase_idx, method='nearest')
    # steepest_increase = df.iloc[[nearest_existing_idx]]

    # Calculate average gas usage before the steepest increase (baseline usage)
    # baseline_avg = df.loc[:steepest_increase.index[0], 'GasUsage'].mean()

    # Calculate heating gas usage by subtracting the baseline average from the total gas usage after the steepest increase
    # df.loc[steepest_increase.index[0]:, 'Heating'] = df.loc[steepest_increase.index[0]:, 'GasUsage'] - baseline_avg
    # df.loc[:steepest_increase.index[0], 'Heating'] = 0

    # Ensure that the heating gas usage values are not negative
    # df['Heating'] = np.maximum(0, df['Heating'])

    # Plot the data, the moving average curve, and mark the steepest increase with a vertical line
    fig, ax = plt.subplots()

    ax.plot(df.index, df['GasUsage'], label='Total GasUsage',color="green")
    # ax.plot(df.index, df['Heating'], label='Heating GasUsage', color='purple')
    # ax.plot(df.index, df['MA'], label='Moving Average', color='red')
    # ax.axvline(steepest_increase.index[0], color='green', linestyle='--', label='Steepest Increase')

    ax.set_title('Gas Usage')
    ax.set_xlabel('Date')
    ax.set_ylabel('Gas Usage')
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
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', )

    plt.show()

file_path = "GasMeter.xlsx"
x = 7  # You can change the value of x to any desired number of days
plot_gas_usage(file_path, x)


