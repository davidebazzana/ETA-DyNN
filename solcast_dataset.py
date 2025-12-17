import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SolcastDataset():

    def __init__(self, path:str):
        self.data = self.read_dataset(path)
        self.dates = self.get_dates()
        self.plot_gti(self.dates[10])
        for t in self.get_data_by_date(self.dates[10]):
            print(f"data: {t}")
    
    
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
        if wind is not None:
            T_mod = T_amb + (G / (U_0 + U_1 * W))
        else:
            T_mod = T_amb + K_T * G

        return T_mod

    
    def compute_reflectivity_effect(self,
                                    G:float,
                                    sun_elevation:float,
                                    sun_azimuth:float,
                                    plane_tilt:float=35,
                                    plane_orientation:float=180):
        """Compute the reflectivity effect of the surface of a PV module.

        Keyword arguments:
        G -- the in-plane irradiance (W/m**2)
        sun_elevation -- the elevation of the sun (deg)
        sun_azimuth -- the azimuth of the sun (deg)
        plane_tilt -- the tilt of the PV module (deg)
        plane_orientation -- the orientation of the PV module (deg)
        """
        pass


    def compute_real_power_output(self,
                                  G:float,
                                  T_mod:float,
                                  G_STC:float=1000,
                                  T_mod_STC:float=25,
                                  k1:float=-0.01724,
                                  k2:float=-0.04047,
                                  k3:float=-0.0047,
                                  k4:float=1.49*(10**-4),
                                  k5:float=1.47*(110**-4),
                                  k6:float=5.0*(110**-6):
        """Compute the real power output of a PV module.

        Keyword arguments:
        G -- the in-plane irradiance (W/m**2), accounted for the reflectivity effect
        T_mod -- the PV module temperature (°C)
        G_STC -- the in-plane irradiance as defined by the Standard Test Conditions (default 1000 W/m**2)
        T_mod_STC -- the PV module temperature as defined by the Standard Test Conditions (default 25°C)
        k1, k2, k3, k4, k5, k6 -- the empirical coefficients of the model
        """
        G_prime = in_plane_irradiance / G_STC
        T_prime = T_mod - T_mod_STC
        P = G_prime * (P_mod_STC +
                       k1 * np.log(G_prime) +
                       k2 * (np.log(G_prime) ** 2) +
                       k3 * T_prime +
                       k4 * T_prime * np.log(G_prime) +
                       k5 * T_prime * (np.log(G_prime) ** 2) +
                       k6 * (T_prime ** 2))

        return P


    def compute_relative_conversion_efficiency(self, P:float, G:float, G_STC:float=1000, T_mod_STC:float=25):
        """Compute the relative conversion efficiency of a PV module.

        Keyword arguments:
        P -- the real power output (W)
        G -- the in-plane irradiance (W/m**2)
        G_STC -- the in-plane irradiance as defined by the Standard Test Conditions (default 1000 W/m**2)
        T_mod_STC -- the PV module temperature as defined by the Standard Test Conditions (default 25°C)
        """
        G_prime = G / G_STC
        eta_rel = P / (P_mod_STC * G_prime)

        return eta_rel


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
