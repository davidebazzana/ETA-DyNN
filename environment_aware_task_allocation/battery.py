import math
import numpy as np
import matplotlib.pyplot as plt


class Battery:
    def __init__(self,
                 battery_capacity: float = 100.0,   # Ah
                 nominal_voltage: float = 12.8,      # V
                 v_max: float = 14.4,                # V
                 v_min: float = 12.0,                # V
                 efficiency: float = 0.98,
                 initial_soc: float = 0.4,
                 discharge_cutoff_limit: float = 0.2,
                 max_charge_rate: float = 0.5,       # C-rate
                 soc_optimal_range: tuple = (0.2, 0.8),
                 rated_charge_cycle_life: int = 1000):
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
        self.v_min = v_min
        self.efficiency = efficiency
        self.max_charge_rate = max_charge_rate
        self.soc_optimal_range = soc_optimal_range

        if discharge_cutoff_limit > initial_soc:
            raise ValueError(f"Initial SOC {initial_soc} is below discharge cutoff {discharge_cutoff_limit}")

        self.initial_soc = initial_soc
        self.discharge_cutoff_limit = discharge_cutoff_limit
        self.soc = initial_soc
        
        self.cutoff_buffer = (
            self.discharge_cutoff_limit
            * self.battery_capacity
            * self.nominal_voltage
        )


        self.rated_charge_cycle_life = rated_charge_cycle_life
        self.discharged_energy_ah = 0.0
        self.n_charge_cycles = 0

    
    def _estimate_voltage(self) -> float:
        # Estiamate internal voltage (LiFePO4-style flat curve)
        if self.soc <= 0.1:
            return self.v_min
        elif self.soc <= 0.9:
            return self.nominal_voltage
        else:
            # Steep rise near full (CV region)
            return self.nominal_voltage + (
                (self.v_max - self.nominal_voltage)
                * (self.soc - 0.9) / 0.1
            )

    def recharge_battery(self, power_in: float, dt: float):
        """Update the battery state of charge. Calculate the potential
        charging current (I = P/V) using a simplified voltage curve
        for Lithium and a rough approximation of LiFePO4 rise. The
        logic used is CC-CV and C-rate clipping.

        Keyword arguments:
        power_in -- the power input provided by the PV module
        dt -- the delta of time recharging (in seconds)
        """
        if power_in <= 0 or self.soc >= 1.0:
            return self._estimate_voltage(), 0.0

        dt_h = dt / 3600.0

        voltage = min(self._estimate_voltage(), self.v_max)

        # CC current from available power
        potential_current = power_in / voltage if voltage > 0 else 0.0

        # C-rate limit
        max_current = self.max_charge_rate * self.battery_capacity
        actual_current = min(potential_current, max_current)

        # CV taper (voltage-driven, not SOC-driven)
        if voltage >= self.v_max:
            taper = max(0.0, (1.0 - self.soc) / 0.1)
            actual_current *= taper

        actual_current = max(0.0, actual_current)

        # Apply charging efficiency
        delta_ah = actual_current * dt_h * self.efficiency
        self.soc += delta_ah / self.battery_capacity
        self.soc = min(self.soc, 1.0)

        return voltage, actual_current

    def compute_harvested_energy(self, power_in: float, dt: float) -> float:
        if math.isnan(power_in):
            power_in = 0.0
        return power_in * (dt / 3600.0)  # Wh

    def available_energy(self) -> float:
        return max(
            self.soc * self.battery_capacity * self.nominal_voltage,
            0.0
        )

    def within_budget(self, energy: float) -> bool:
        return self.available_energy() - self.cutoff_buffer >= energy

    def how_many_within_budget(self, individual_energy_cost: float):
        return int((self.available_energy() - self.cutoff_buffer) / individual_energy_cost)

    def use(self, energy: float) -> float:
        """
        energy : requested energy in Wh
        """
        available_energy_wh = self.available_energy()
        energy_used_wh = min(energy, available_energy_wh)
        
        remaining_energy_wh = available_energy_wh - energy_used_wh
        self.soc = max(
            remaining_energy_wh
            / (self.battery_capacity * self.nominal_voltage),
            0.0
        )

        discharged_ah = energy_used_wh / self.nominal_voltage
        self.discharged_energy_ah += discharged_ah

        if self.discharged_energy_ah >= self.battery_capacity:
            self.discharged_energy_ah -= self.battery_capacity
            self.n_charge_cycles += 1

        return energy_used_wh

    def wh_to_ah(self, wh: float) -> float:
        return wh / self.nominal_voltage

    def ah_to_wh(self, ah: float) -> float:
        return ah * self.nominal_voltage

    def reset_log(self):
        self.n_charge_cycles = 0
    
    def reset(self):
        self.soc = self.initial_soc
        self.discharged_energy_ah = 0.0

        self.reset_log()

    def set_soc(self, soc:float):
        self.soc = soc

if __name__ == "__main__":

    hours = np.linspace(0, 12, 1000)
    dt_hours = hours[1] - hours[0]
    dt = dt_hours * 3600

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
