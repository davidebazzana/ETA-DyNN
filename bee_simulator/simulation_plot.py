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
from summary_plot import SummaryPlot
from solcast_dataset import SolcastDataset
import numpy as np

def retrieve_returned_by(returned_by_e0,
                         returned_by_e1,
                         returned_by_vit4v):
    returned_by = np.zeros_like(returned_by_e0, dtype=int)
    returned_by[returned_by_e1] = 1
    returned_by[returned_by_vit4v] = 2

    return returned_by

class SimulationPlot():
    def __init__(self, sim:bool=True):
        self.sim = sim

        self.fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(3, 2)

        self.ax_irradiance = self.fig.add_subplot(gs[:, 0])
        self.ax_returned_by = self.fig.add_subplot(gs[0, 1])
        self.ax_classification_performance = self.fig.add_subplot(gs[1, 1])
        self.ax_performance = self.fig.add_subplot(gs[2, 1])

    def set_simulation(self,
                       returned_by_e0,
                       returned_by_e1,
                       returned_by_vit4v,
                       global_answers,
                       vit4v_answers,
                       labels,
                       dt:int=10):
        returned_by = retrieve_returned_by(returned_by_e0,
                                           returned_by_e1,
                                           returned_by_vit4v)

        print(f"{len(labels)=}")
        print(f"Infested: {sum(labels)}, Free: {len(labels) - sum(labels)}")

        dataset = "./solcast_dataset_202408_sassari.json"
        solcast = SolcastDataset(dataset,
                                 dt=dt,
                                 P_mod_STC=125)
        
        dataset = Dataset(returned_by, global_answers, vit4v_answers, labels)
        self.experiment = Experiment(dataset=dataset,
                                     solcast=solcast,
                                     battery_capacity=200,
                                     memory_capacity=5000,
                                     initial_soc=0.4,
                                     dt=dt,
                                     future_time_window=360,
                                     stage_energy_cost=0.25, # 0.25
                                     delegation_energy_cost=0.1, # 0.1
                                     idle_energy_cost=0.01, # 0.01
                                     force_delegation=True)

        ts, gtis_logs, power_outputs_logs, recovery_state_logs, returned_by_logs, returned_by_none_logs, battery_logs, memory_logs, decisions_logs, discount_logs, stage_cost_logs, delegation_cost_logs, egress_rate_logs, new_tasks_logs, dropped_tasks_logs, bsi, ere, pdm, hm = self.experiment.launch()
        print(f"{bsi=}")
        print(f"{ere=}")
        print(f"{pdm=}")
        print(f"{hm=}")

        print(f"{np.sum(decisions_logs == -1)=}")
        print(f"{np.sum(decisions_logs == 0)=}")
        print(f"{np.sum(decisions_logs == 1)=}")

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
        self.ax_classification_performance.clear()
        
        self.ax_irradiance.plot(ts, gtis_logs_mean, 'b-', label="GTI")
        self.ax_irradiance.fill_between(ts,
                                        gtis_logs_mean - gtis_logs_std,
                                        gtis_logs_mean + gtis_logs_std,
                                        alpha=0.3)
        self.ax_irradiance.set_xlabel("time step")
        self.ax_irradiance.set_ylabel("W/m²")

        self.ax_egress_rate = self.ax_irradiance.twinx()
        self.ax_egress_rate.plot(ts, egress_rate_logs_mean, "r-", label="Egress rate")
        self.ax_egress_rate.set_ylabel('cnt/min')
        self.ax_egress_rate.tick_params(axis='y')

        memory_line, = self.ax_performance.plot(ts, memory_logs_mean, '-', label='memory')
        self.ax_performance.fill_between(ts,
                                         memory_logs_mean - memory_logs_std,
                                         memory_logs_mean + memory_logs_std,
                                         alpha=0.3)
        battery_line, = self.ax_performance.plot(ts, battery_logs_mean, '-', label='battery')
        self.ax_performance.fill_between(ts,
                                         battery_logs_mean - battery_logs_std,
                                         battery_logs_mean + battery_logs_std,
                                         alpha=0.3)
        
        self.ax_returned_by.stackplot(ts,
                                      returned_by_logs_0_mean,
                                      returned_by_logs_1_mean,
                                      returned_by_logs_2_mean,
                                      dropped_tasks_logs_mean,
                                      labels=['returned by exit 0', 'returned by exit 1', 'returned by vit4v', 'dropped'])

        self.ax_classification_performance.plot(ts, discount_logs_mean, '-', label='discount')
        self.ax_classification_performance.plot(ts, stage_cost_logs_mean, '-', label='stage_cost')
        self.ax_classification_performance.plot(ts, delegation_cost_logs_mean, '-', label='delegation_cost')
        """
        accuracy_line, = self.ax_classification_performance.plot(ts, accuracy_logs_mean, '-', label='accuracy')
        precision_line, = self.ax_classification_performance.plot(ts, precision_logs_mean, '-', label='precision')
        recall_line, = self.ax_classification_performance.plot(ts, recall_logs_mean, '-', label='recall')
        f1_score_line, = self.ax_classification_performance.plot(ts, f1_score_logs_mean, '-', label='f1-score')
        """
        """
        self.ax_classification_performance.fill_between(ts,
                                                        [max(m - accuracy_logs_std[idx], 0) for idx, m in enumerate(accuracy_logs_mean)],
                                                        [min(m + accuracy_logs_std[idx], 1) for idx, m in enumerate(accuracy_logs_mean)],
                                                        alpha=0.3)
        self.ax_classification_performance.fill_between(ts,
                                                        [max(m - precision_logs_std[idx], 0) for idx, m in enumerate(precision_logs_mean)],
                                                        [min(m + precision_logs_std[idx], 1) for idx, m in enumerate(precision_logs_mean)],
                                                        alpha=0.3)
        self.ax_classification_performance.fill_between(ts,
                                                        [max(m - recall_logs_std[idx], 0) for idx, m in enumerate(recall_logs_mean)],
                                                        [min(m + recall_logs_std[idx], 1) for idx, m in enumerate(recall_logs_mean)],
                                                        alpha=0.3)
        self.ax_classification_performance.fill_between(ts,
                                                        [max(m - f1_score_logs_std[idx], 0) for idx, m in enumerate(f1_score_logs_mean)],
                                                        [min(m + f1_score_logs_std[idx], 1) for idx, m in enumerate(f1_score_logs_mean)],
                                                        alpha=0.3)
        """

        """
        self.agent_lines = {'memory_log': memory_line,
                            'battery_log': battery_line,
                            'accuracy_log': accuracy_line,
                            'precision_log': precision_line,
                            'recall_log': recall_line,
                            'f1_score_log': f1_score_line}
        """
        
        self.ax_performance.legend(loc='best', frameon=False)
        self.ax_performance.autoscale()
        
        self.ax_returned_by.legend(loc='best', frameon=False)
        self.ax_returned_by.autoscale()

        self.ax_classification_performance.legend(loc='best', frameon=False)
        self.ax_classification_performance.autoscale()

        self.fig.canvas.draw_idle()

    def get_mean_std(self, data):
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)

        return data_mean, data_std

    def get_classification_performance_mean(self, data):
        # Remove all rows where all the elements are -1
        d = np.copy(data)
        means = np.array([])
        res = np.full(d.shape[1], np.nan)

        while np.sum(d != -1) > 0:
            cond = d == -1
            n_cols = d.shape[1]
            I = np.ones((n_cols, 1))
            cond = cond @ I < n_cols
            d = d[cond.squeeze(), :]

            cond = d == -1
            n_rows = d.shape[0]
            I = np.ones(n_rows)
            cond = I @ cond != 0
            mean = d[:, ~cond].mean(axis=0)
            means = np.insert(means, 0, mean)
            d = d[:, cond]
    
        res[-len(means):] = means
        return res
    
    def update_state(self):
        try:
            data = next(self.experiment)
            
            self.ax_irradiance.plot(data["timestamps"], data["power_outputs"], '-')
            self.ax_irradiance.set_xlabel("Time (HH:MM)")
            self.ax_irradiance.set_ylabel("W")

            self.update_stackplot(data)
            for attr, line in self.agent_lines.items():
                line.set_xdata(ts)
                data = getattr(self.agent, attr)
                line.set_ydata(data)
            self.ax_classification_performance.relim()
            self.ax_classification_performance.autoscale_view()
            self.ax_performance.relim()
            self.ax_performance.autoscale_view()
            
            self.fig.canvas.draw_idle()

        except StopIteration:
            pass

    def update_stackplot(self, data):
        self.ax_returned_by.cla()
        self.ax_returned_by.stackplot(data["timestamps"],
                                      self.experiment.agent.returned_by_log[0],
                                      self.experiment.agent.returned_by_log[1],
                                      self.experiment.agent.returned_by_log[2],
                                      labels=['returned by exit 0', 'returned by exit 1', 'returned by vit4v'])
        self.ax_returned_by.legend()
