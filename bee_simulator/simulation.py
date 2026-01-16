import pickle
import numpy as np

from solcast_dataset import SolcastDataset
from environment_aware_task_allocation.agent import Agent
from environment_aware_task_allocation.data import Dataset
from environment_aware_task_allocation.battery import Battery
from environment_aware_task_allocation.memory import Memory
from environment_aware_task_allocation.experiment import Experiment
from environment_aware_task_allocation.utils import daily_solar_irradiance


class Simulation():
    def __init__(self, end_of_simulation_callback):
        self.end_of_simulation_callback = end_of_simulation_callback

    def set_simulation_data(self,
                            returned_by_e0,
                            returned_by_e1,
                            returned_by_vit4v,
                            global_answers,
                            vit4v_answers,
                            labels,
                            dt:int=10):
        self.dt = dt
        
        returned_by = self.retrieve_returned_by(returned_by_e0,
                                                returned_by_e1,
                                                returned_by_vit4v)
        
        self.dataset = Dataset(returned_by, global_answers, vit4v_answers, labels)
        # Write answers dataset to file
        with open("experiment_samples_dataset.pkl", "wb") as f:
            pickle.dump(self.dataset, f)

    def compare_hardware(self):
        n = 8
        module_powers = np.linspace(10, 500, n, dtype=int)
        battery_capacities = np.linspace(10, 500, n, dtype=int)
        memory_capacities = np.linspace(100, 10_000, n, dtype=int)

        data = []
        for p in tqdm(module_powers):
            for b in battery_capacities:
                for m in memory_capacities:
                    solcast_dataset_path = "./solcast_dataset_202408_sassari.json"
                    solcast = SolcastDataset(solcast_dataset_path,
                                             dt=self.dt,
                                             P_mod_STC=p) # 125
        
                    self.experiment = Experiment(dataset=self.dataset,
                                                 solcast=solcast,
                                                 battery_capacity=b, # 200
                                                 memory_capacity=m, #5000
                                                 initial_soc=0.5,
                                                 dt=self.dt,
                                                 future_time_window=360,
                                                 stage_energy_cost=0.25, # 0.25
                                                 delegation_energy_cost=0.1, # 0.1
                                                 idle_energy_cost=0.01, # 0.01
                                                 force_delegation=False)

                    ts, gtis_logs, power_outputs_logs, recovery_state_logs, returned_by_logs, returned_by_none_logs, battery_logs, memory_logs, decisions_logs, discount_logs, stage_cost_logs, delegation_cost_logs, egress_rate_logs, new_tasks_logs, dropped_tasks_logs, bsi, ere, pdm, hm = self.experiment.launch()
                    data.append(np.array([p, b, m, bsi, ere, pdm, hm, dropped_tasks_logs[0][-1]]))
        data = np.array(data)
        print(f"{data=}")
        with open("simulation_data.pkl", "wb") as f:
            pickle.dump(data, f)


    def launch_simulation(self):        
        solcast_dataset_path = "./solcast_2024_sassari.json" # "./solcast_dataset_202408_sassari.json"
        solcast = SolcastDataset(solcast_dataset_path,
                                 dt=self.dt,
                                 P_mod_STC=150) # 125
        
        self.experiment = Experiment(dataset=self.dataset,
                                     solcast=solcast,
                                     battery_capacity=400, #200
                                     memory_capacity=4000, #5000
                                     initial_soc=0.5,
                                     dt=self.dt,
                                     future_time_window=360,
                                     stage_energy_cost=0.25, # 0.25
                                     delegation_energy_cost=0.1, # 0.1
                                     idle_energy_cost=0.01, # 0.01
                                     force_delegation=False)

        self.data_log = self.experiment.launch()

        self.end_of_simulation_callback(self.data_log)

    
    def retrieve_returned_by(self,
                             returned_by_e0,
                             returned_by_e1,
                             returned_by_vit4v):
        returned_by = np.zeros_like(returned_by_e0, dtype=int)
        returned_by[returned_by_e1] = 1
        returned_by[returned_by_vit4v] = 2

        return returned_by
