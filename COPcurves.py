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
colors = ['blue', 'purple', 'red']  # Color scheme for cold to hot

fig, ax = plt.subplots()

for indoor_temp, color in zip(indoor_temps, colors):
    if indoor_temp == 20:
        outdoor_temps_range = np.linspace(-20, 10, 100)
    else:
        outdoor_temps_range = outdoor_temps

    cop_values = [calculate_cop(outdoor_temp, indoor_temp) for outdoor_temp in outdoor_temps_range]
    ax.plot(outdoor_temps_range, cop_values, label=f'{indoor_temp}°C', color=color)

ax.set_xlabel('Cold Temperature (°C)', fontsize=14)
ax.set_ylabel('COP', fontsize=14)
ax.legend(title='Indoor Temperature', fontsize=12, title_fontsize=12)
ax.set_title('Theoretical COP values', fontsize=16)
ax.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.tight_layout()

plt.show()
