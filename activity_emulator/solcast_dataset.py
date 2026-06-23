from tqdm import tqdm
from typing import Literal
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date as datetime_date
from datetime import datetime, timedelta
from environment_aware_task_allocation.battery import Battery

logger = logging.getLogger(__name__)

class SolcastDataset():

    def __init__(self,
                 path:str,
                 dt:float=300,
                 P_mod_STC:float=40,
                 ref_day:datetime_date=datetime_date(2024,8,2),
                 compute_ghi_correlation:bool=False,
                 partition:str="validation"):
        self.data = self.read_dataset(path)
        self.dates = self.get_dates()
        self.ref_day = ref_day

        self.valid_dates = []
        self.interesting_dates = []
        gtis = []
        ghis = []
        power_outputs = []
        timestamps = []

        for date in tqdm(self.dates, desc=f"Scanning the dataset {path}"):
            gti, _ = self.retrieve_data(date, "gti")
            ghi, _ = self.retrieve_data(date, "ghi")

            if len(gti) != 288 or len(ghi) != 288:
                """Check if the date has a valid number of data points."""
                continue

            self.valid_dates.append(date)

        if compute_ghi_correlation: self.compute_ghi_correlation()

        self.valid_dates = np.array(self.valid_dates)
        if partition == "train":
            self.valid_dates = self.valid_dates[181:188]
        """
        elif partition == "validation":
            self.valid_dates = self.valid_dates[0:31]
        """

        for date in tqdm(self.valid_dates, desc="Retrieving data"):
            self.interesting_dates.append(date)
            solar_zenith, _ = self.retrieve_data(date, "zenith")
            solar_azimuth, _ = self.retrieve_data(date, "azimuth")
            gti, _ = self.retrieve_data(date, "gti")
            ghi, _ = self.retrieve_data(date, "ghi")
            T_amb, ts = self.retrieve_data(date, "air_temp")
            
            real_p = self.compute_power_output(G=gti,
                                               T_amb=T_amb,
                                               solar_zenith=solar_zenith,
                                               solar_azimuth=solar_azimuth,
                                               P_mod_STC=P_mod_STC)
            gtis.append(gti)
            ghis.append(ghi)
            power_outputs.append(real_p)
            timestamps.append(ts)
        self.gtis = np.array(gtis)
        self.ghis = np.array(ghis)
        self.power_outputs = np.array(power_outputs)
        self.timestamps = np.array(timestamps)

        resampled_timestamps = []
        for ts in self.timestamps:
            resampled_timestamps.append(self.resample_datetimes(ts, dt))
        resampled_timestamps = np.array(resampled_timestamps)

        resampled_gtis = []
        resampled_ghis = []
        resampled_power_outputs = []
        for new_day, gtis, ghis, power_outputs, timestamps in zip(resampled_timestamps, self.gtis, self.ghis, self.power_outputs, self.timestamps):
            new_day = np.linspace(0, 1, len(new_day))
            timestamps = np.linspace(0, 1, len(timestamps))
            resampled_gtis.append(np.interp(new_day, timestamps, gtis))
            resampled_ghis.append(np.interp(new_day, timestamps, ghis))
            resampled_power_outputs.append(np.interp(new_day, timestamps, power_outputs))
        resampled_gtis = np.array(resampled_gtis)
        resampled_ghis = np.array(resampled_ghis)
        resampled_power_outputs = np.array(resampled_power_outputs)

        self.day = 0
        self.timestamps = resampled_timestamps
        self.gtis = resampled_gtis
        self.ghis = resampled_ghis
        self.power_outputs = resampled_power_outputs

    
    def __iter__(self):
        self.day = 0
        return self


    def __next__(self):
        if self.day == len(self.timestamps):
            raise StopIteration
        day = self.interesting_dates[self.day]
        ts = self.timestamps[self.day]
        gtis = self.gtis[self.day]
        ghis = self.ghis[self.day]
        power_outputs = self.power_outputs[self.day]
        self.day += 1
        return {
            "date": day,
            "timestamps": ts,
            "gtis": gtis,
            "ghis": ghis,
            "power_outputs": power_outputs
        }

    def split_indices(self, total_length, block_size, val_fraction=0.8, seed=None):
        """
        Split indices from 0 to total_length-1 into non-overlapping blocks of size block_size.
    
        Args:
        total_length (int): Total number of indices (e.g., 363).
        block_size (int): Size of each contiguous block.
        val_fraction (float): Fraction of blocks to use for validation.
        seed (int, optional): Random seed for reproducibility.

        Returns:
        train_idx (np.ndarray): Array of training indices.
        validation_idx (np.ndarray): Array of validation indices.
        """
        if seed is not None:
            np.randnom.seed(seed)
            
        # Create blocks of contiguous indices
        blocks = [np.arange(i, min(i + block_size, total_length)) 
                  for i in range(0, total_length, block_size)]
    
        # Shuffle the blocks
        np.random.shuffle(blocks)
    
        # Split into train and validation
        n_val = int(len(blocks) * val_fraction)
        val_blocks = blocks[:n_val]
        train_blocks = blocks[n_val:]
    
        # Flatten the lists of blocks
        train_idx = np.concatenate(train_blocks)
        validation_idx = np.concatenate(val_blocks)
    
        return train_idx, validation_idx

    def _correlation(self, sig1:np.array, sig2:np.array):
        assert len(sig1) == len(sig2), f"The two signals have different lengths: {len(sig1)}, {len(sig2)}"

        return np.sum(sig1 * sig2)

    def correlation(self, day:np.array):
        ref_ghi, _ = self.retrieve_data(self.ref_day, "ghi")
        other_ghi, _ = self.retrieve_data(day, "ghi")
        ref_ghi = np.array(ref_ghi)
        other_ghi = np.array(other_ghi)
        
        return self._correlation(ref_ghi, other_ghi) / np.sqrt(self._correlation(ref_ghi, ref_ghi) * self._correlation(other_ghi, other_ghi))

    def compute_ghi_correlation(self, show_plot:bool=False):
        correlations = []
        highlight_indexes = []
        for idx, date in enumerate(self.valid_dates):
            correlation = self.correlation(date)
            # print(f"{date} correlation to {self.ref_day}: {correlation}")

            correlations.append((correlation, date))

            if (date == datetime_date(2024,3,3) or
                date == datetime_date(2024,11,5) or
                date == datetime_date(2024,8,14) or
                date == datetime_date(2024,8,2)):
                highlight_indexes.append(idx)
        n_min = 10
        min_correlation_dates = sorted(correlations, key=lambda x: x[0])[:n_min]

        sorted_correlations = sorted(correlations, key=lambda x: x[0])
        mean_correlation_date = sorted_correlations[len(sorted_correlations)//2]
        highest_correlation_date = sorted_correlations[-2]
        ref_day_date = sorted_correlations[-1]

        if show_plot:
            correlations = np.array(correlations)

            plt.scatter(
                [correlations[:, 1][i] for i in highlight_indexes],  # x-values to highlight
                [correlations[:, 0][i] for i in highlight_indexes],  # y-values to highlight
                s=100,          # size of the circle
                facecolors='none',  # hollow circle
                edgecolors='red',   # circle color
                linewidths=1.5,     # thickness of circle
                zorder=5
            )
        
            plt.plot(correlations[:, 1], correlations[:, 0],
                     color='green',
                     marker='o',
                     linestyle='',
                     markersize=5,
                     zorder=2)

            plt.xlabel("Date")
            plt.ylabel(f"GHI Correlation with {self.ref_day}")

            # plt.ylim(0, 1)
            plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
            
            plt.legend()
            
            plt.show()


        return min_correlation_dates, mean_correlation_date, highest_correlation_date, ref_day_date

    
    def resample_datetimes(self, timestamps: np.ndarray, delta_t: float) -> np.ndarray:
        """
        Parameters
        ----------
        timestamps : np.ndarray of datetime.datetime
        Input timestamps (order does not matter).
        delta_t : float
        Desired time step in seconds.
        
        Returns
        -------
        np.ndarray of datetime.datetime
        Evenly spaced timestamps with spacing delta_t.
        """
        if timestamps.size == 0:
            return np.array([], dtype=object)

        # Ensure sorted timestamps
        timestamps = np.sort(timestamps)

        start = timestamps[0]
        end = timestamps[-1]
        
        step = timedelta(seconds=delta_t)

        result = []
        current = start
        while current <= end:
            result.append(current)
            current += step

        return np.array(result, dtype=object)

    
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
                             P_mod_STC:float=40,
                             show_plot:bool=False,
                             time:np.array=None):
        incidence = self.compute_angle_of_incidence(solar_zenith,
                                                    solar_azimuth,
                                                    tilt,
                                                    azimuth)
        G_AL = self.compute_angular_losses(incidence, G)
        T_mod = self.compute_module_temperature(G_AL, T_amb, W=W)
        real_p = self.compute_real_power_output(G_AL, T_mod, P_mod_STC=P_mod_STC)

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


    def retrieve_data(self, date, data_type:Literal["zenith", "azimuth", "ghi", "gti", "air_temp", "wind"]):
        """ Retrieve the request data for the given date.

        Keyword arguments:
        date -- the date for which to retrieve data for.
        """
        if date not in self.dates:
            raise ValueError(f"Date {date} is not in the dataset")
        data = self.get_data_by_date(date)
        res = []
        for t in data:
            if data_type in ["zenith", "azimuth"]:
                res.append(np.deg2rad(t[data_type]))
            else:
                res.append(t[data_type])

        time = np.array([datetime.fromisoformat(d["period_end"]) for d in data])

        return res, time


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
    
    dataset = "./solcast_2024_sassari.json" # "./solcast_dataset_202408_sassari.json"
    solcast = SolcastDataset(path=dataset,
                             dt=10,
                             P_mod_STC=80,
                             plot_correlation=True)

    """
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
        
    battery = Battery()
    for p_solar in real_p:
        battery.recharge_battery(p_solar, dt)
        soc_history.append(battery.soc * 100)

    plt.figure(figsize=(12, 6))
    
    plt.subplot()
    plt.plot(x, soc_history, color='green')
    plt.title("Battery State of Charge (%)")
    plt.xlabel("Hours")
    plt.ylabel("SoC %")

    plt.tight_layout()
    plt.show()
    """

    """
    gtis = []
    real_ps = []
    soc_histories = []
    for date in solcast.dates:
        if date.day == 1 or date.day == 31:
            continue

        solar_zenith, _ = solcast.retrieve_data(date, "zenith")
        solar_azimuth, _ = solcast.retrieve_data(date, "azimuth")
        gti, _ = solcast.retrieve_data(date, "gti")
        T_amb, x = solcast.retrieve_data(date, "air_temp")

        real_p = solcast.compute_power_output(G=gti,
                                              T_amb=T_amb,
                                              solar_zenith=solar_zenith,
                                              solar_azimuth=solar_azimuth,
                                              show_plot=False)
        gtis.append(gti)
        real_ps.append(real_p)
        soc_history = []
        
        dt = (x[1]-x[0]).total_seconds() / 3600
        
        battery = Battery()
        for p_solar in real_p:
            battery.recharge_battery(p_solar, dt)
            soc_history.append(battery.soc * 100)

        soc_histories.append(soc_history)

    gtis = np.array(gtis)
    real_ps = np.array(real_ps)
    soc_histories = np.array(soc_histories)

    gtis_mean = gtis.mean(axis=0)
    gtis_std = gtis.std(axis=0)

    real_ps_mean = real_ps.mean(axis=0)
    real_ps_std = real_ps.std(axis=0)

    soc_histories_mean = soc_histories.mean(axis=0)
    soc_histories_std = soc_histories.std(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    line1, = ax1.plot(x, gtis_mean, label="GTI")
    ax1.set_xlabel("Time (HH:MM)")
    ax1.set_ylabel("W/m^2")
    ax1.fill_between(x, gtis_mean - gtis_std, gtis_mean + gtis_std, alpha=0.3)
    ax1_2 = ax1.twinx()
    line2, = ax1_2.plot(x, real_ps_mean, linestyle="--", label="Power Output")
    ax1_2.set_ylabel("W")
    ax1_2.fill_between(x, real_ps_mean - real_ps_std, real_ps_mean + real_ps_std, alpha=0.3)
            
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels)
    
    ax2.plot(x, soc_histories_mean, label="State of Charge")
    ax2.set_xlabel("Time (HH:MM)")
    ax2.set_ylabel("%")
    ax2.fill_between(x, soc_histories_mean - soc_histories_std, soc_histories_mean + soc_histories_std, alpha=0.3)
    ax2.legend()

    fig.autofmt_xdate()
    plt.show()

    """
