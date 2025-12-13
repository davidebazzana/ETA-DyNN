import cProfile
import pstats
from pstats import SortKey
import time
from environment_aware_task_allocation.functions import Agent
from environment_aware_task_allocation.utils import daily_solar_irradiance
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from summary_plot import SummaryPlot
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
                       labels):
        returned_by = retrieve_returned_by(returned_by_e0,
                                           returned_by_e1,
                                           returned_by_vit4v)
        
        self.agent = Agent(returned_by = returned_by,
                           answers = global_answers,
                           labels = labels,
                           max_battery=50,
                           max_memory=50,
                           initial_battery=25,
                           initial_memory=0,
                           n_stages = 2,
                           stage_energy_cost=0.25, # 0.1,
                           delegation_energy_cost=0.1, # 0.05,
                           gamma_s_1=0.7,
                           gamma_s_2=0.3,
                           gamma_d_1=0.9,
                           gamma_d_2=0.2)
        
        self.t_data = []

        self.ax_irradiance.clear()
        self.ax_performance.clear()
        self.ax_returned_by.clear()
        self.ax_classification_performance.clear()
        
        self.ax_irradiance.plot(self.t_data, [], '-')
        self.ax_irradiance.set_xlabel("time step")
        self.ax_irradiance.set_ylabel("W/m²")
        
        memory_line, = self.ax_performance.plot(self.t_data, [], '-', label='memory')
        battery_line, = self.ax_performance.plot(self.t_data, [], '-', label='battery')

        self.ax_returned_by.stackplot(self.t_data, [], [], [], labels=['returned by exit 0', 'returned by exit 1', 'returned by vit4v'])
        
        accuracy_line, = self.ax_classification_performance.plot(self.t_data, [], '-', label='accuracy')
        precision_line, = self.ax_classification_performance.plot(self.t_data, [], '-', label='precision')
        recall_line, = self.ax_classification_performance.plot(self.t_data, [], '-', label='recall')
        f1_score_line, = self.ax_classification_performance.plot(self.t_data, [], '-', label='f1-score')

        self.agent_lines = {'memory_log': memory_line,
                            'battery_log': battery_line,
                            'accuracy_log': accuracy_line,
                            'precision_log': precision_line,
                            'recall_log': recall_line,
                            'f1_score_log': f1_score_line}

        self.ax_performance.legend(loc='best', frameon=False)
        self.ax_performance.autoscale()
        
        self.ax_returned_by.legend(loc='best', frameon=False)
        self.ax_returned_by.autoscale()

        self.ax_classification_performance.legend(loc='best', frameon=False)
        self.ax_classification_performance.autoscale()

        self.t_data.append(0)

        self.agent_it = iter(self.agent)
        
    def update_state(self):
        try:
            t = next(self.agent)
            
            self.t_data.append(t)

            self.ax_irradiance.plot(self.t_data, [daily_solar_irradiance(t_i) for t_i in self.t_data], '-')
            self.ax_irradiance.set_xlabel("time step")
            self.ax_irradiance.set_ylabel("W/m²")

            self.update_stackplot()
            for attr, line in self.agent_lines.items():
                line.set_xdata(self.t_data)
                data = getattr(self.agent, attr)
                line.set_ydata(data)
            self.ax_classification_performance.relim()
            self.ax_classification_performance.autoscale_view()
            self.ax_performance.relim()
            self.ax_performance.autoscale_view()
            
            self.fig.canvas.draw_idle()

        except StopIteration:
            pass

    def update_stackplot(self):
        self.ax_returned_by.cla()
        self.ax_returned_by.stackplot(self.t_data,
                                      self.agent.returned_by_log[0],
                                      self.agent.returned_by_log[1],
                                      self.agent.returned_by_log[2],
                                      labels=['returned by exit 0', 'returned by exit 1', 'returned by vit4v'])
        self.ax_returned_by.legend()
