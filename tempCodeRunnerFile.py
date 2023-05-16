import numpy as np
import matplotlib.pyplot as plt

def calculate_cop(outdoor_temp, indoor_temp):
    # Calculate COP using the maximum theoretical efficiency formula
    Tc = outdoor_temp + 273.15
    Th = indoor_temp + 273.15
    cop_heating = Th / (Th - Tc)
    return cop_heating

outdoor_temps = np.linspace(-20, 20, 100)
indoor_temps = [20, 30, 40]

for indoor_temp in indoor_temps:
    if indoor_temp == 20:
        outdoor_temps_range = np.linspace(-20, 10, 100)
    else:
        outdoor_temps_range = outdoor_temps

    cop_values = [calculate_cop(outdoor_temp, indoor_temp) for outdoor_temp in outdoor_temps_range]
    plt.plot(outdoor_temps_range, cop_values, label=f'{indoor_temp}°C')

plt.xlabel('Outdoor Temperature (°C)')
plt.ylabel('COP')
plt.legend(title='Hot Temperature')
plt.title('COP vs. Outdoor Temperature for Different Indoor Temperatures')
plt.grid()
plt.show()
