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


def launch_simulation_worker(args):
    cls, init_soc, p, b, m, gamma_s_1, gamma_s_2, gamma_d_1, gamma_d_2, device = args
    result = cls.launch_simulation(
        P_mod_STC=p,
        battery_initial_soc=init_soc,
        battery_capacity=b,
        memory_capacity=m,
        gamma_s_1=gamma_s_1,
        gamma_s_2=gamma_s_2,
        gamma_d_1=gamma_d_1,
        gamma_d_2=gamma_d_2,
        force_delegation=False,
        daily_reset=False,
        device=device,
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
                            energy_performance,
                            dt:int=10):
        self.dt = dt
        
        returned_by = self.retrieve_returned_by(returned_by_e0,
                                                returned_by_e1,
                                                returned_by_vit4v)
        
        self.dataset = Dataset(scores, returned_by, global_answers, vit4v_answers, labels, energy_performance)
        # Write answers dataset to file
        with open("experiment_samples_dataset.pkl", "wb") as f:
            pickle.dump(self.dataset, f)

    def load_simulation_data(self, path:str="experiment_samples_dataset.pkl"):
        with open(path, "rb") as f:
            self.dataset = pickle.load(f)
            
    def compare_hardware(self,
                         gamma_s_1:float,
                         gamma_s_2:float,
                         gamma_d_1:float,
                         gamma_d_2:float,
                         device:str,
                         battery_initial_soc:int=50):
        hardware_dims = {
            "titan": {
                "module_powers": {
                    "min": 100,
                    "max": 700
                },
                "battery_capacities": {
                    "min": 20,
                    "max": 500
                },
                "memory_capacities": {
                    "min": 20,
                    "max": 5_000
                }
            },
            "jetson": {
                "module_powers": {
                    "min": 10,
                    "max": 70
                },
                "battery_capacities": {
                    "min": 20,
                    "max": 500
                },
                "memory_capacities": {
                    "min": 20,
                    "max": 5_000
                }
            }
        }
        """
        n = 4
        module_powers = np.linspace(20, 500, n, dtype=int)
        battery_capacities = np.linspace(20, 500, n, dtype=int)
        memory_capacities = np.linspace(20, 5_000, n, dtype=int)
        """
        n = 4
        module_powers = np.linspace(hardware_dims[device]["module_powers"]["min"],
                                    hardware_dims[device]["module_powers"]["max"], n, dtype=int)
        battery_capacities = np.linspace(hardware_dims[device]["battery_capacities"]["min"],
                                         hardware_dims[device]["battery_capacities"]["max"], n, dtype=int)
        memory_capacities = np.linspace(hardware_dims[device]["memory_capacities"]["min"],
                                        hardware_dims[device]["memory_capacities"]["max"], n, dtype=int)

        # print(f"{module_powers=}")
        # print(f"{battery_capacities=}")
        # print(f"{memory_capacities=}")
        tasks = [(self, battery_initial_soc, p, b, m, gamma_s_1, gamma_s_2, gamma_d_1, gamma_d_2, device) for p, b, m in itertools.product(module_powers, battery_capacities, memory_capacities)]

        results = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for res in executor.map(launch_simulation_worker, tasks):
                results.append(res)
                print(f"[ProcessPoolExecutor] {len(results)=}")
        
        with open("hardware_comparison_results_4.pkl", "wb") as f:
            pickle.dump(results, f)
        

    def launch_simulation(self,
                          P_mod_STC:int,
                          battery_initial_soc:int,
                          battery_capacity:int,
                          memory_capacity:int,
                          gamma_s_1:float,
                          gamma_s_2:float,
                          gamma_d_1:float,
                          gamma_d_2:float,
                          force_delegation:bool,
                          daily_reset:bool,
                          device:str,
                          end_callback=None,
                          solcast_partition:str="validation"):
        battery_initial_soc = battery_initial_soc / 100
        print("======= Simulation =======")
        print(f"{P_mod_STC=}")
        print(f"{battery_initial_soc=}")
        print(f"{battery_capacity=}")
        print(f"{memory_capacity=}")
        print(f"{gamma_s_1=}")
        print(f"{gamma_s_2=}")
        print(f"{gamma_d_1=}")
        print(f"{gamma_d_2=}")
        print(f"{force_delegation=}")
        print(f"{daily_reset=}")
        print(f"{device=}")

        solcast_dataset_path = "./solcast_2024_sassari.json" # "./solcast_dataset_202408_sassari.json"
        solcast = SolcastDataset(solcast_dataset_path,
                                 dt=self.dt,
                                 P_mod_STC=P_mod_STC,
                                 partition=solcast_partition) # 125

        # Costs in Wh (assuming 10 seconds idle time)

        energy_costs = {
            "titan": {
                "stage": 0.3539396298521146,
                "idle": 0.2687317150290075,
                "delegation": 0.056629709442887354
            },
            "jetson": {
                "stage": 0.06556391602883709,
                "idle": 0.016505677295133903,
                "delegation": 0.056629709442887354
            }
        }
        
        self.experiment = Experiment(dataset=self.dataset,
                                     solcast=solcast,
                                     battery_capacity=battery_capacity, #200
                                     memory_capacity=memory_capacity, #5000
                                     initial_soc=battery_initial_soc,
                                     dt=self.dt,
                                     future_time_window=360,
                                     stage_energy_cost=energy_costs[device]["stage"], # 0.25
                                     delegation_energy_cost=energy_costs[device]["delegation"], # 0.1
                                     idle_energy_cost=energy_costs[device]["idle"], # 0.01
                                     force_delegation=force_delegation,
                                     daily_reset=daily_reset,
                                     gamma_s_1=gamma_s_1,
                                     gamma_s_2=gamma_s_2,
                                     gamma_d_1=gamma_d_1,
                                     gamma_d_2=gamma_d_2)

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
    
