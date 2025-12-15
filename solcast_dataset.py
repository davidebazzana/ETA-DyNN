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
