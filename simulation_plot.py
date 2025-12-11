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

    def start_simulation(self,
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

        steps = range(self.agent.DAILY_STEPS)
        self.ax_irradiance.plot(steps, [daily_solar_irradiance(t) for t in steps], '-')
        self.ax_irradiance.set_xlabel("time step")
        self.ax_irradiance.set_ylabel("W/m²")
        
        t_data = []
        
        memory_line, = self.ax_performance.plot(t_data, [], '-', label='memory')
        battery_line, = self.ax_performance.plot(t_data, [], '-', label='battery')

        """
        returned_by_e0_line, = self.ax_returned_by.plot(t_data, [], '-', label='returned by exit 0')
        returned_by_e1_line, = self.ax_returned_by.plot(t_data, [], '-', label='returned by exit 1')
        returned_by_vit4v_line, = self.ax_returned_by.plot(t_data, [], '-', label='returned by vit4v')
        """
        self.ax_returned_by.stackplot(t_data, [], [], [], labels=['returned by exit 0', 'returned by exit 1', 'returned by vit4v'])
        
        accuracy_line, = self.ax_classification_performance.plot(t_data, [], '-', label='accuracy')
        precision_line, = self.ax_classification_performance.plot(t_data, [], '-', label='precision')
        recall_line, = self.ax_classification_performance.plot(t_data, [], '-', label='recall')
        f1_score_line, = self.ax_classification_performance.plot(t_data, [], '-', label='f1-score')
        """
        completed_tasks_local_line, = self.ax_performance.plot(t_data, [], '-', label='local')
        completed_tasks_remote_line, = self.ax_performance.plot(t_data, [], '-', label='remote')
        dropped_tasks_line, = self.ax_performance.plot(t_data, [], '-', label='dropped')
        recovery_state_line, = self.ax_performance.plot(t_data, [], '-', label='recovery')
        """

        """
        agent_data = {'memory': ([], memory_line),
                      'battery': ([], battery_line),
                      'completed_tasks_local_log': ([], completed_tasks_local_line),
                      'completed_tasks_remote_log': ([], completed_tasks_remote_line),
                      'dropped_tasks_log': ([], dropped_tasks_line),
                      'recovery_state_log': ([], recovery_state_line)}
        """
        agent_lines = {'memory_log': memory_line,
                      'battery_log': battery_line,
                      'accuracy_log': accuracy_line,
                      'precision_log': precision_line,
                      'recall_log': recall_line,
                      'f1_score_log': f1_score_line}

        self.ax_performance.legend(loc='best', frameon=False)
        self.ax_performance.set_xlim(0, self.agent.DAILY_STEPS)
        # self.ax_performance.set_ylim(0, 10)
        
        self.ax_returned_by.legend(loc='best', frameon=False)
        self.ax_returned_by.set_xlim(0, self.agent.DAILY_STEPS)

        self.ax_classification_performance.legend(loc='best', frameon=False)
        self.ax_classification_performance.set_xlim(0, self.agent.DAILY_STEPS)

        t_data.append(0)

        for t in self.agent:
            t_data.append(t)

            data_min_max = {
                "returned_by": {
                    "min": 0,
                    "max": 0
                },
                "classification_performance": {
                    "min": 0,
                    "max": 0
                },
                "performance": {
                    "min": 0,
                    "max": 0
                }
            }
            attr_data_map = {
                "memory_log": "performance",
                "battery_log": "performance",
                "accuracy_log": "classification_performance",
                "precision_log": "classification_performance",
                "recall_log": "classification_performance",
                "f1_score_log": "classification_performance"
            }
            self.update_stackplot(t_data)
            for attr, line in agent_lines.items():
                """
                if attr == "returned_by_log":
                    data = {
                        0: [],
                        1: [],
                        2: []
                    }
                    for idx, returned_by_line in enumerate(line):
                        
                        returned_by_line.set_xdata(t_data)
                        data = getattr(self.agent, attr)[idx]
                                                
                        returned_by_line.set_ydata(data)
                        data_min_max["returned_by"]["min"] = min(data_min_max["returned_by"]["min"],
                                                                 min(data))
                        data_min_max["returned_by"]["max"] = max(data_min_max["returned_by"]["max"],
                                                                 max(data))
                else:
                """
                line.set_xdata(t_data)
                data = getattr(self.agent, attr)
                line.set_ydata(data)
                data_min_max[attr_data_map[attr]]["min"] = min(data_min_max[attr_data_map[attr]]["min"],
                                                               min(data))
                data_min_max[attr_data_map[attr]]["max"] = max(data_min_max[attr_data_map[attr]]["max"],
                                                               max(data))

            # self.ax_returned_by.set_ylim(data_min_max["returned_by"][]["min"], data_min_max["returned_by"]["max"])
            self.ax_classification_performance.set_ylim(data_min_max["classification_performance"]["min"], data_min_max["classification_performance"]["max"])
            self.ax_performance.set_ylim(data_min_max["performance"]["min"], data_min_max["performance"]["max"])

        if self.canvas is not None:
            self.canvas.draw()

    def update_stackplot(self, t):
        self.ax_returned_by.cla()
        self.ax_returned_by.stackplot(t,
                                      self.agent.returned_by_log[0],
                                      self.agent.returned_by_log[1],
                                      self.agent.returned_by_log[2],
                                      labels=['returned by exit 0', 'returned by exit 1', 'returned by vit4v'])
        self.ax_returned_by.legend()

    def set_canvas(self, canvas):
        self.canvas = canvas
