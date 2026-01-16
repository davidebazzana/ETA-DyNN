import pickle
import sys
import cProfile
import pstats
from pstats import SortKey
import time
from environment_aware_task_allocation.agent import Agent
from environment_aware_task_allocation.data import Dataset
from environment_aware_task_allocation.battery import Battery
from environment_aware_task_allocation.memory import Memory
from environment_aware_task_allocation.experiment import Experiment
from environment_aware_task_allocation.utils import daily_solar_irradiance
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from summary_plot import SummaryPlot
from solcast_dataset import SolcastDataset
import numpy as np
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from tqdm import tqdm


class SimulationPlot():
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(2, 2)

        self.ax_irradiance = self.fig.add_subplot(gs[:, 0])
        self.ax_returned_by = self.fig.add_subplot(gs[0, 1])
        # self.ax_classification_performance = self.fig.add_subplot(gs[1, 1])
        self.ax_performance = self.fig.add_subplot(gs[1, 1])

    def plot(self,
             classification,
             ts,
             gtis_logs,
             power_outputs_logs,
             recovery_state_logs,
             returned_by_logs,
             returned_by_none_logs,
             battery_logs,
             memory_logs,
             decisions_logs,
             discount_logs,
             stage_cost_logs,
             delegation_cost_logs,
             egress_rate_logs,
             new_tasks_logs,
             dropped_tasks_logs,
             bsi,
             ere,
             pdm,
             hm,
             energy_del,
             energy_dem,
             energy_harv):
        print("==============================")
        print(f"{classification['acc']=}, {classification['prec']=}, {classification['rec']=}, {classification['f1']=}")
        print(f"No decision (either m=0 or recovery state): {np.sum(decisions_logs == -1)}")
        print(f"Decisions to delegate: {np.sum(decisions_logs ==  0)}")
        print(f"Decisions to compute locally: {np.sum(decisions_logs ==  1)}")
        print(f"{energy_del=}, {energy_dem=}, {energy_harv=}")
        print(f"{bsi=}, {ere=}, {pdm=}, {hm=}")
        print("==============================")
        
        print(f"{bsi=}")
        print(f"{ere=}")
        print(f"{pdm=}")
        print(f"{hm=}")

        print(f"{np.sum(decisions_logs == -1)=}")
        print(f"{np.sum(decisions_logs == 0)=}")
        print(f"{np.sum(decisions_logs == 1)=}")

        print(f"{dropped_tasks_logs[0][-1]=}")

        gtis_logs_mean, gtis_logs_std = self.get_mean_std(gtis_logs)
        power_outputs_logs_mean, power_outputs_logs_std = self.get_mean_std(power_outputs_logs)
        recovery_state_logs_mean, recovery_state_logs_std = self.get_mean_std(recovery_state_logs)
        returned_by_logs_0_mean, returned_by_logs_0_std = self.get_mean_std(returned_by_logs[0])
        returned_by_logs_1_mean, returned_by_logs_1_std = self.get_mean_std(returned_by_logs[1])
        returned_by_logs_2_mean, returned_by_logs_2_std = self.get_mean_std(returned_by_logs[2])
        returned_by_none_logs_mean, returned_by_none_logs_std = self.get_mean_std(returned_by_none_logs)
        battery_logs_mean, battery_logs_std = self.get_mean_std(battery_logs)
        memory_logs_mean, memory_logs_std = self.get_mean_std(memory_logs)
        egress_rate_logs_mean, egress_rate_logs_std = self.get_mean_std(egress_rate_logs)
        new_tasks_logs_mean, new_tasks_logs_std = self.get_mean_std(new_tasks_logs)
        dropped_tasks_logs_mean, dropped_tasks_logs_std = self.get_mean_std(dropped_tasks_logs)
        discount_logs_mean, discount_logs_std = self.get_mean_std(discount_logs)
        stage_cost_logs_mean, stage_cost_logs_std = self.get_mean_std(stage_cost_logs)
        delegation_cost_logs_mean, delegation_cost_logs_std = self.get_mean_std(delegation_cost_logs)

        self.ax_irradiance.clear()
        self.ax_performance.clear()
        self.ax_returned_by.clear()
        # self.ax_classification_performance.clear()
        
        self.ax_irradiance.plot(ts, gtis_logs_mean, 'b-', label="GTI")
        self.ax_irradiance.fill_between(ts,
                                        gtis_logs_mean - gtis_logs_std,
                                        gtis_logs_mean + gtis_logs_std,
                                        alpha=0.3)
        self.ax_irradiance.set_xlabel("Time (HH:MM)")
        self.ax_irradiance.set_ylabel("W/m²")

        self.ax_egress_rate = self.ax_irradiance.twinx()
        self.ax_egress_rate.plot(ts, egress_rate_logs_mean, "r-", label="Egress rate")
        self.ax_egress_rate.set_ylabel('cnt/min')
        self.ax_egress_rate.tick_params(axis='y')

        memory_line, = self.ax_performance.plot(ts, memory_logs_mean, '-', label='Memory Usage')
        self.ax_performance.fill_between(ts,
                                         memory_logs_mean - memory_logs_std,
                                         memory_logs_mean + memory_logs_std,
                                         alpha=0.3)
        battery_line, = self.ax_performance.plot(ts, battery_logs_mean, '-', label='Battery SOC')
        self.ax_performance.fill_between(ts,
                                         battery_logs_mean - battery_logs_std,
                                         battery_logs_mean + battery_logs_std,
                                         alpha=0.3)
        self.ax_performance.set_xlabel("Time (HH:MM)")
        self.ax_performance.set_ylabel("Normalized")
        
        self.ax_returned_by.stackplot(ts,
                                      returned_by_logs_0_mean,
                                      returned_by_logs_1_mean,
                                      returned_by_logs_2_mean,
                                      dropped_tasks_logs_mean,
                                      labels=['returned by exit 0', 'returned by exit 1', 'returned by vit4v', 'dropped'])
        self.ax_returned_by.set_ylabel("Tasks")
        # self.ax_returned_by.set_xlabel("Time (HH:MM)")

        self.ax_performance.set_ylim(0, 1)
        self.ax_performance.legend(loc='best', frameon=False)        
        self.ax_irradiance.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ax_returned_by.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ax_performance.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        self.ax_returned_by.legend(loc='upper left', frameon=False)
        self.ax_returned_by.autoscale()

        self.fig.canvas.draw_idle()

    def get_mean_std(self, data):
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)

        return data_mean, data_std
        
    def save_plots(self):
        self.fig.savefig("simulation_plot_results.pdf", format='pdf', bbox_inches='tight')
