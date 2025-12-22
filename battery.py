import numpy as np
import matplotlib.pyplot as plt

class Battery():

    def __init__(self,
                 battery_capacity:float=100.0,
                 nominal_voltage:float=12.8,
                 v_max:float=14.4,
                 efficiency:float=0.98,
                 initial_soc:float=0.2,
                 max_charge_rate:float=0.5):
        """
        Keyword arguments:
        battery_capacity -- total capacity (Amp-hours)
        nominal_voltage -- nominal voltage (Volts)
        v_max -- absorption/cut-off voltage
        efficiency -- charging efficiency
        max_charge_rate -- maximum charging rate (50A for 100Ah battery)
        """
        self.battery_capacity = battery_capacity
        self.nominal_voltage = nominal_voltage
        self.v_max = v_max
        self.efficiency = efficiency
        self.max_charge_rate = max_charge_rate

        self.soc = initial_soc
        self.energy_stored = self.soc * self.battery_capacity * self.nominal_voltage

    def recharge_battery(self, power_in:float, dt:float):
        """Update the battery state of charge. Calculate the potential
        charging current (I = P/V) using a simplified voltage curve
        for Lithium and a rough approximation of LiFePO4 rise. The
        logic used is CC-CV and C-rate clipping.

        """
        current_v = self.nominal_voltage + (self.soc * 1.4)
        if self.soc > 0.95:
            current_v = self.v_max
    
        potential_current = power_in / current_v
    
        actual_current = min(potential_current, self.max_charge_rate * self.battery_capacity)
    
        if current_v >= self.v_max:
            actual_current = actual_current * (1 - self.soc) / 0.05 
        
        actual_current = max(0, actual_current)
        delta_ah = actual_current * dt
        self.soc += delta_ah / self.battery_capacity
    
        if self.soc > 1.0:
            self.soc = 1.0
            actual_current = 0

        return current_v, actual_current

    def use_battery(self, energy:float):
        pass
    

if __name__ == "__main__":

    hours = np.linspace(0, 12, 1000)
    dt = hours[1] - hours[0]

    peak_solar_power = 600 
    p_solar_all = peak_solar_power * np.sin(np.pi * hours / 12)

    soc_history = []
    v_history = []
    p_in_history = []

    battery = Battery()
    for p_solar in p_solar_all:
        current_v, actual_current = battery.recharge_battery(p_solar, dt)
        
        soc_history.append(battery.soc * 100)
        v_history.append(current_v)
        p_in_history.append(actual_current * current_v)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(hours, p_solar_all, label="Solar Output (W)", linestyle='--')
    plt.plot(hours, p_in_history, label="Power into Battery (W)", color='orange')
    plt.title("Power Generation vs Intake")
    plt.xlabel("Hours")
    plt.ylabel("Watts")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hours, soc_history, color='green')
    plt.title("Battery State of Charge (%)")
    plt.xlabel("Hours")
    plt.ylabel("SoC %")

    plt.tight_layout()
    plt.show()
