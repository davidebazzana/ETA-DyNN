import pickle
import numpy as np
from datetime import date as datetime_date
import threading
import itertools
from concurrent.futures import ProcessPoolExecutor
import os

from solcast_dataset import SolcastDataset
from environment_aware_task_allocation.agent import Agent
from environment_aware_task_allocation.data import Dataset
from environment_aware_task_allocation.battery import Battery
from environment_aware_task_allocation.memory import Memory
from environment_aware_task_allocation.experiment import Experiment
from environment_aware_task_allocation.utils import daily_solar_irradiance

from hardware_comparison_db_utils import insert_experiment


def launch_simulation_worker(args):
    cls, init_soc, p, b, m = args
    result = cls.launch_simulation(
        P_mod_STC=p,
        battery_initial_soc=init_soc,
        battery_capacity=b,
        memory_capacity=m,
        start_date=datetime_date.today(),
        end_date=datetime_date.today(),
        force_delegation=False,
        daily_reset=False,
        solcast_partition="train"
    )

    return {
        "P_mod_STC": int(p),
        "battery": int(b),
        "memory": int(m),
        "results": result
    }


class Simulation():
    def __init__(self, dt:int=10):
        self.dt = dt

    def set_simulation_data(self,
                            scores,
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
        
        self.dataset = Dataset(scores, returned_by, global_answers, vit4v_answers, labels)
        # Write answers dataset to file
        with open("experiment_samples_dataset.pkl", "wb") as f:
            pickle.dump(self.dataset, f)

    def load_simulation_data(self, path:str="experiment_samples_dataset.pkl"):
        with open(path, "rb") as f:
            self.dataset = pickle.load(f)
            
    def compare_hardware(self,
                         battery_initial_soc:float=0.5):
        n = 4
        init_soc = 50
        module_powers = np.linspace(20, 500, n, dtype=int)
        battery_capacities = np.linspace(20, 500, n, dtype=int)
        memory_capacities = np.linspace(20, 5_000, n, dtype=int)

        tasks = [(self, init_soc, p, b, m) for p, b, m in itertools.product(module_powers, battery_capacities, memory_capacities)]

        results = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for res in executor.map(launch_simulation_worker, tasks):
                results.append(res)
                print(f"[ProcessPoolExecutor] {len(results)=}")
        
        with open("hardware_comparison_results_3.pkl", "wb") as f:
            pickle.dump(results, f)
        

    def launch_simulation(self,
                          P_mod_STC:int,
                          battery_initial_soc:int,
                          battery_capacity:int,
                          memory_capacity:int,
                          force_delegation:bool,
                          daily_reset:bool,
                          end_callback=None,
                          solcast_partition:str="validation"):
        battery_initial_soc = battery_initial_soc / 100
        print("======= Simulation =======")
        print(f"{P_mod_STC=}")
        print(f"{battery_initial_soc=}")
        print(f"{battery_capacity=}")
        print(f"{memory_capacity=}")
        print(f"{start_date=}")
        print(f"{end_date=}")
        print(f"{force_delegation=}")
        print(f"{daily_reset=}")

        solcast_dataset_path = "./solcast_2024_sassari.json" # "./solcast_dataset_202408_sassari.json"
        solcast = SolcastDataset(solcast_dataset_path,
                                 dt=self.dt,
                                 P_mod_STC=P_mod_STC,
                                 partition=solcast_partition) # 125
        
        self.experiment = Experiment(dataset=self.dataset,
                                     solcast=solcast,
                                     battery_capacity=battery_capacity, #200
                                     memory_capacity=memory_capacity, #5000
                                     initial_soc=battery_initial_soc,
                                     dt=self.dt,
                                     future_time_window=360,
                                     stage_energy_cost=0.25, # 0.25
                                     delegation_energy_cost=0.1, # 0.1
                                     idle_energy_cost=0.01, # 0.01
                                     force_delegation=force_delegation,
                                     daily_reset=daily_reset)

        self.data_log = self.experiment.launch()

        if end_callback is not None:
            end_callback(self.data_log)

        return self.data_log

    def retrieve_returned_by(self,
                             returned_by_e0,
                             returned_by_e1,
                             returned_by_vit4v):
        returned_by = np.zeros_like(returned_by_e0, dtype=int)
        returned_by[returned_by_e1] = 1
        returned_by[returned_by_vit4v] = 2

        return returned_by


if __name__ == "__main__":
    sim = Simulation()
    sim.load_simulation_data()
    sim.compare_hardware()
    
