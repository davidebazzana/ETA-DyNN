import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from battery import Battery

logger = logging.getLogger(__name__)

class SolcastDataset():

    def __init__(self, path:str):
        self.data = self.read_dataset(path)
        self.dates = self.get_dates()

    
    def read_dataset(self, path):
        """Read the dataset."""
        with open(path, 'r') as fp:
            data = json.load(fp)["estimated_actuals"]
        return data

    
    def get_dates(self):
        """Retrieve all the available dates in the dataset self.data."""
        dates = set()
        for d in self.data:
            dates.add(datetime.fromisoformat(d["period_end"]).date())
        dates = sorted([d for d in dates])
        return dates


    def get_data_by_date(self, date:datetime.date):
        """Retrieve the data for a given date."""
        return [
            s for s in self.data
            if datetime.fromisoformat(s["period_end"]).date() == date
        ]


    def compute_module_temperature(self,
                                   G:float,
                                   T_amb:float,
                                   W:float|None,
                                   U_0:float=26.9,
                                   U_1:float=6.2,
                                   K_T:float=0.035):
        """Compute the temperature of a PV module.

        Keyword arguments:
        G -- the in-plane irradiance (W/m**2)
        T_amb -- the ambient temperature (°C)
        W -- the wind (m/s)
        U_0, U_1 -- coefficients to compute the module temperature accounting for wind
        K_T -- coefficient to compute the module temperature without accounting for the wind
        """
        if W is not None:
            T_mod = T_amb + (G / (U_0 + U_1 * W))
        else:
            T_mod = T_amb + K_T * G

        return T_mod

    
    def compute_angular_losses(self,
                               alpha:float,
                               G:float|None=None,
                               a_r:float=0.159):
        """Compute the angular losses of a PV module.

        Keyword arguments:
        alpha -- tha angle of incidence of the Sun's beam radiation on the PV module
        G -- the in-plane irradiance (W/m**2)
        a_r -- the angular losses coefficient
        """
        AL = 1 - ((1 - np.exp((-np.cos(alpha)) / a_r)) / (1 - np.exp(-1 / a_r)))
        if G is not None:
            return np.maximum(G - (G * AL), 0)
        else:
            return AL

    
    def compute_real_power_output(self,
                                  G:float,
                                  T_mod:float,
                                  G_STC:float=1000,
                                  T_mod_STC:float=25,
                                  P_mod_STC:float=40,
                                  k1:float=-0.01724,
                                  k2:float=-0.04047,
                                  k3:float=-0.0047,
                                  k4:float=1.49*(10**-4),
                                  k5:float=1.47*(110**-4),
                                  k6:float=5.0*(110**-6)):
        """Compute the real power output of a PV module.

        Keyword arguments:
        G -- the in-plane irradiance (W/m**2), accounted for the reflectivity effect
        T_mod -- the PV module temperature (°C)
        G_STC -- the in-plane irradiance as defined by the Standard Test Conditions (default 1000 W/m**2)
        T_mod_STC -- the PV module temperature as defined by the Standard Test Conditions (default 25°C)
        P_mod_STC -- the maximum power point of the PV module at STC (W)
        k1, k2, k3, k4, k5, k6 -- the empirical coefficients of the model
        """
        G_prime = G / G_STC
        T_prime = T_mod - T_mod_STC
        P = G_prime * (P_mod_STC +
                       k1 * np.log(G_prime) +
                       k2 * (np.log(G_prime) ** 2) +
                       k3 * T_prime +
                       k4 * T_prime * np.log(G_prime) +
                       k5 * T_prime * (np.log(G_prime) ** 2) +
                       k6 * (T_prime ** 2))

        return P


    def compute_relative_conversion_efficiency(self,
                                               P:float,
                                               G:float,
                                               G_STC:float=1000,
                                               T_mod_STC:float=25,
                                               P_mod_STC:float=40):
        """Compute the relative conversion efficiency of a PV module.

        Keyword arguments:
        P -- the real power output (W)
        G -- the in-plane irradiance (W/m**2)
        G_STC -- the in-plane irradiance as defined by the Standard Test Conditions (default 1000 W/m**2)
        T_mod_STC -- the PV module temperature as defined by the Standard Test Conditions (default 25°C)
        P_mod_STC -- the maximum power point of the PV module at STC (W)
        """
        G_prime = G / G_STC
        eta_rel = P / (P_mod_STC * G_prime)

        return eta_rel


    def compute_angle_of_incidence(self,
                                   solar_zenith:float,
                                   solar_azimuth:float,
                                   tilt:float=np.deg2rad(35),
                                   azimuth:float=np.deg2rad(180)):
        """Compute the angle of incidence of the Sun's beam radiation and the PV module."""
        aoi = np.arccos(np.cos(solar_zenith)*np.cos(tilt) + np.sin(solar_zenith)*np.sin(tilt)*np.cos(solar_azimuth - azimuth))

        return aoi


    def compute_power_output(self,
                             G:float,
                             T_amb:float,
                             solar_zenith:float,
                             solar_azimuth:float,
                             tilt:float=np.deg2rad(35),
                             azimuth:float=np.deg2rad(180),
                             W:float|None=None,
                             show_plot:bool=False,
                             time:np.array=None):
        incidence = self.compute_angle_of_incidence(solar_zenith,
                                                    solar_azimuth,
                                                    tilt,
                                                    azimuth)
        G_AL = self.compute_angular_losses(incidence, G)
        T_mod = self.compute_module_temperature(G_AL, T_amb, W=W)
        real_p = self.compute_real_power_output(G_AL, T_mod)

        if show_plot:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

            line1, = ax1.plot(time, G, label="GTI")
            line2, = ax1.plot(time, G_AL, label="GTI w AL")
            ax1.set_xlabel("Time (HH:MM)")
            ax1.set_ylabel("W/m^2")
            ax1_2 = ax1.twinx()
            line3, = ax1_2.plot(time, incidence, linestyle="--", label="Incidence")
            ax1_2.set_ylabel("rad")
            
            lines = [line1, line2, line3]
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels)
            

            ax2.plot(time, T_amb, label="Ambient Temperature")
            ax2.plot(time, T_mod, label="PV Module Temperature")
            ax2.set_xlabel("Time (HH:MM)")
            ax2.set_ylabel("C°")
            ax2.legend()

            line1, = ax3.plot(time, G, label="gti")
            ax3.set_xlabel("Time (HH:MM)")
            ax3.set_ylabel("W/m^2")

            ax3_2 = ax3.twinx()
            line2, = ax3_2.plot(time, real_p, linestyle="--", label="power output")
            ax3_2.set_ylabel("W")

            start = time[0].replace(hour=0, minute=0)
            end = start + timedelta(days=1)
            ax3.set_xlim(start, end)

            ticks = [start + timedelta(hours=h) for h in range(0, 25, 2)]
            ax3.set_xticks(ticks)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            lines = [line1, line2]
            labels = [line.get_label() for line in lines]
            ax3.legend(lines, labels)

            fig.autofmt_xdate()
            plt.show()

        return real_p


    def retrieve_useful_data(self, date):
        if date not in self.dates:
            raise ValueError(f"Date {date} is not in the dataset")
        data = self.get_data_by_date(date)
        solar_zenith = []
        solar_azimuth = []
        gti = []
        T_amb = []
        for t in data:
            solar_zenith.append(np.deg2rad(t["zenith"]))
            solar_azimuth.append(np.deg2rad(t["azimuth"]))
            gti.append(t["gti"])
            T_amb.append(t["air_temp"])
        solar_zenith = np.array(solar_zenith)
        solar_azimuth = np.array(solar_azimuth)
        gti = np.array(gti)
        T_amb = np.array(T_amb)

        time = np.array([datetime.fromisoformat(d["period_end"]) for d in data])

        return solar_zenith, solar_azimuth, gti, T_amb, time


    def plot_gti(self, date:datetime.date):
        date_data = self.get_data_by_date(date)
        # fractional hours
        x = np.array([datetime.fromisoformat(d["period_end"]) for d in date_data])
        y = np.array([d["gti"] for d in date_data])

        fig, ax = plt.subplots()

        ax.plot(x, y)

        start = x[0].replace(hour=0, minute=0)
        end = start + timedelta(days=1)
        ax.set_xlim(start, end)

        ticks = [start + timedelta(hours=h) for h in range(0, 25, 2)]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        ax.set_xlabel("Time (HH:MM)")
        ax.set_ylabel("GTI")
        ax.set_title("Global Titled Irradiance")

        fig.autofmt_xdate()
        plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    dataset = "./solcast_dataset_202408_sassari.json"
    solcast = SolcastDataset(dataset)

    solcast.plot_gti(solcast.dates[10])

    solar_zenith, solar_azimuth, gti, T_amb, x = solcast.retrieve_useful_data(solcast.dates[10])

    alphas = np.linspace(0, np.pi/2, 400)
    ALs = []
    for i in alphas:
        ALs.append(solcast.compute_angular_losses(i))
    plt.figure()
    plt.plot(alphas, ALs)
    plt.xlabel("Incidences")
    plt.ylabel("Angular Losses")
    plt.title("Plot of Angular Losses on [0, π/2]")
    plt.grid(True)
    plt.show()

    aoi = solcast.compute_angle_of_incidence(solar_zenith=solar_zenith,
                                             solar_azimuth=solar_azimuth)

    y_z = np.rad2deg(np.pi/2 - solar_zenith)
    y_a = np.rad2deg(solar_azimuth)
    y_aoi = np.rad2deg(aoi)
    fig, ax = plt.subplots()

    ax.plot(x, y_z, label="solar zenith")
    ax.plot(x, y_a, label="solar azimuth")
    ax.plot(x, y_aoi, label="angle of incidence")

    start = x[0].replace(hour=0, minute=0)
    end = start + timedelta(days=1)
    ax.set_xlim(start, end)

    ticks = [start + timedelta(hours=h) for h in range(0, 25, 2)]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
    ax.set_xlabel("Time (HH:MM)")
    ax.set_ylabel("Degrees")
    ax.legend()

    fig.autofmt_xdate()
    plt.show()

    real_p = solcast.compute_power_output(G=gti,
                                          T_amb=T_amb,
                                          solar_zenith=solar_zenith,
                                          solar_azimuth=solar_azimuth,
                                          show_plot=True,
                                          time=x)
        
    soc_history = []

    dt = (x[1]-x[0]).total_seconds() / 3600
    print(f"{dt=}")
        
    battery = Battery()
    for p_solar in real_p:
        battery.recharge_battery(p_solar, dt)
        soc_history.append(battery.soc * 100)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 2)
    plt.plot(x, soc_history, color='green')
    plt.title("Battery State of Charge (%)")
    plt.xlabel("Hours")
    plt.ylabel("SoC %")

    plt.tight_layout()
    plt.show()
