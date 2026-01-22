from activity_emulator.solcast_dataset import SolcastDataset
from environment_aware_task_allocation.agent import Agent
from environment_aware_task_allocation.data import Dataset
from environment_aware_task_allocation.battery import Battery
from environment_aware_task_allocation.memory import Memory
from tqdm import tqdm
import numpy as np
from datetime import datetime, date, timedelta

from environment_aware_task_allocation.results import Results, ClassificationPerformances


class Experiment():

    def __init__(self,
                 dataset:Dataset,
                 solcast:SolcastDataset,
                 battery_capacity:float=100,
                 memory_capacity:int=500,
                 initial_soc:float=0.4,
                 dt:float=10,
                 future_time_window:int=5,
                 stage_energy_cost:float=0.25,
                 delegation_energy_cost:float=0.1,
                 idle_energy_cost:float=0.01,
                 force_delegation:bool=False,
                 daily_reset:bool=False):
        battery = Battery(battery_capacity=battery_capacity,
                          initial_soc=initial_soc,
                          max_charge_rate=0.5)
        memory = Memory(max_tasks=memory_capacity)
        self.solcast = solcast
        self.agent = Agent(battery=battery,
                           memory=memory,
                           dataset=dataset,
                           dt=dt,
                           n_stages=2,
                           future_time_window=future_time_window,
                           stage_energy_cost=stage_energy_cost,
                           delegation_energy_cost=delegation_energy_cost,
                           idle_energy_cost=idle_energy_cost,
                           gamma_s_1=0.5, # 0.7,
                           gamma_s_2=0.5, # 0.3,
                           gamma_d_1=0.5, # 0.9,
                           gamma_d_2=0.5) # 0.2)
        self.force_delegation = force_delegation
        self.daily_reset = daily_reset

    def launch_days(self, data):
        for idx, _ in tqdm(enumerate(day_data["timestamps"])):
            self.agent.compute(
                ghi=data["ghis"][idx],
                power_output=data["power_outputs"][idx],
                ghis_predictions=ghis[idx:idx+self.agent.future_time_window],
                force_delegation=self.force_delegation)
            bsi = self.agent.bsi_log[-1]
            bsi_logs.append(bsi)
            ere_factor_logs.append(self.agent.ere_factor)
            ere = (1 / (day + 1)) * sum(ere_factor_logs)
            ere_logs.append(ere)
            pdm_factor_logs.append(self.agent.pdm_factor)
            pdm = (1 / (day + 1)) * sum(pdm_factor_logs)
            pdm_logs.append(pdm)

        
    def launch(self):
        ts = None
        days = []
        gtis_logs = []
        power_outputs_logs = []

        returned_by_logs = {
            0: [],
            1: [],
            2: []
        }
        returned_by_none_logs = []

        recovery_state_logs = []

        acc_log = []
        prec_log = []
        rec_log = []
        f1_log = []
        
        battery_logs = []
        memory_logs = []

        decisions_logs = []
        discount_logs = []
        stage_cost_logs = []
        delegation_cost_logs = []

        egress_rate_logs = []
        new_tasks_logs = []
        dropped_tasks_logs = []

        energy_del_logs = []
        energy_dem_logs = []
        energy_harv_logs = []

        t_log = []
        t_in_opt_range_log = []
        battery_charge_cycles_log = []
        ere_factor_log = []
        pdm_factor_log = []

        prev_date = datetime.utcfromtimestamp(0).date()
        for day, data in enumerate(self.solcast):
            print(f"DAY {day + 1} -- {data['date']}")
            if ts is None:
                ts = data["timestamps"]
            else:
                for i, t in enumerate(data["timestamps"]):
                    assert ts[i].time() == t.time(), f"Non-homogeneous timestamps: {ts[i].time()} and {t.time()}"
            yesterday = data['date'] - timedelta(days=1)
            if yesterday != prev_date:
                print(f"Yesterday {yesterday} is different than the last day {prev_date}, resetting the agent") 
                self.agent.reset()
            prev_date = data['date']
            
            days.append(data['date'])
            gtis_logs.append(data["gtis"])
            power_outputs_logs.append(data["power_outputs"])

            ghis = np.append(data["ghis"], self.agent.future_time_window*[0])
            for idx, _ in tqdm(enumerate(data["timestamps"])):
                self.agent.compute(
                    ghi=data["ghis"][idx],
                    power_output=data["power_outputs"][idx],
                    ghis_predictions=ghis[idx:idx+self.agent.future_time_window],
                    force_delegation=self.force_delegation)
                
            recovery_state_logs.append(self.agent.recovery_state_log[1:])
                
            returned_by_logs[0].append(self.agent.returned_by_log[0][1:])
            returned_by_logs[1].append(self.agent.returned_by_log[1][1:])
            returned_by_logs[2].append(self.agent.returned_by_log[2][1:])
            returned_by_none_logs.append(self.agent.returned_by_none_log[1:])
            battery_logs.append(self.agent.battery_log[1:])
            memory_logs.append(self.agent.memory_log[1:])
            decisions_logs.append(self.agent.decision_log[1:])
            discount_logs.append(self.agent.discount_log[1:])
            stage_cost_logs.append(self.agent.stage_cost_log[1:])
            delegation_cost_logs.append(self.agent.delegation_cost_log[1:])

            """
            bsi = self.agent.bsi_log[-1]
            bsi_logs.append(bsi)
            ere_factor_logs.append(self.agent.ere_factor)
            ere = (1 / (day + 1)) * sum(ere_factor_logs)
            ere_logs.append(ere)
            pdm_factor_logs.append(self.agent.pdm_factor)
            pdm = (1 / (day + 1)) * sum(pdm_factor_logs)
            pdm_logs.append(pdm)
            hm = (bsi * ere * pdm) ** (1 / 3)
            hm_logs.append(hm)
            """
            t_in_opt_range, t, battery_charge_cycles, ere_factor, pdm_factor = self.agent.get_sustainability_performance()
            t_in_opt_range_log.append(t_in_opt_range)
            t_log.append(t)
            battery_charge_cycles_log.append(battery_charge_cycles)
            ere_factor_log.append(ere_factor)
            pdm_factor_log.append(pdm_factor)

            egress_rate_logs.append(self.agent.egress_rate_log[1:])
            new_tasks_logs.append(self.agent.new_tasks_log[1:])
            dropped_tasks_logs.append(self.agent.dropped_tasks_log[1:])

            acc, prec, rec, f1 = self.agent.get_classification_performance()
            acc_log.append(acc)
            prec_log.append(prec)
            rec_log.append(rec)
            f1_log.append(f1)
            energy_del_logs.append(self.agent.energy_delivered_log)
            energy_dem_logs.append(self.agent.energy_demand_log)
            energy_harv_logs.append(self.agent.energy_harvested_log)

            if self.daily_reset:
                self.agent.reset()
            else:
                self.agent.reset_logs()
            
        ts = np.array(ts)
        days = np.array(days)
        gtis_logs = np.array(gtis_logs)
        power_outputs_logs = np.array(power_outputs_logs)
        recovery_state_logs = np.array(recovery_state_logs)
        acc_log = np.array(acc_log)
        prec_log = np.array(prec_log)
        rec_log = np.array(rec_log)
        f1_log = np.array(f1_log)
        returned_by_logs = {
            0: np.array(returned_by_logs[0]),
            1: np.array(returned_by_logs[1]),
            2: np.array(returned_by_logs[2])
        }
        returned_by_none_logs = np.array(returned_by_none_logs)
        battery_logs = np.array(battery_logs)
        memory_logs = np.array(memory_logs)
        decisions_logs = np.array(decisions_logs)
        discount_logs = np.array(discount_logs)
        stage_cost_logs = np.array(stage_cost_logs)
        delegation_cost_logs = np.array(delegation_cost_logs)
        egress_rate_logs = np.array(egress_rate_logs)
        new_tasks_logs = np.array(new_tasks_logs)
        dropped_tasks_logs = np.array(dropped_tasks_logs)
        energy_del_logs = np.array(energy_del_logs)
        energy_dem_logs = np.array(energy_dem_logs)
        energy_harv_logs = np.array(energy_harv_logs)
        t_log = np.array(t_log)
        t_in_opt_range_log = np.array(t_in_opt_range_log)
        battery_charge_cycles_log = np.array(battery_charge_cycles_log)
        ere_factor_log = np.array(ere_factor_log)
        pdm_factor_log = np.array(pdm_factor_log)

        """
        data = {
            "days": days,
            "classification": {
                "acc": acc_log,
                "prec": prec_log,
                "rec": rec_log,
                "f1": f1_log
            },
            "ts": ts,
            "gtis_logs": gtis_logs,
            "power_outputs_logs": power_outputs_logs,
            "recovery_state_logs": recovery_state_logs,
            "returned_by_logs": returned_by_logs,
            "returned_by_none_logs": returned_by_none_logs,
            "battery_logs": battery_logs,
            "memory_logs": memory_logs,
            "decisions_logs": decisions_logs,
            "discount_logs": discount_logs,
            "stage_cost_logs": stage_cost_logs,
            "delegation_cost_logs": delegation_cost_logs,
            "egress_rate_logs": egress_rate_logs,
            "new_tasks_logs": new_tasks_logs,
            "dropped_tasks_logs": dropped_tasks_logs,
            "energy_del_logs": energy_del_logs,
            "energy_dem_logs": energy_dem_logs,
            "energy_harv_logs": energy_harv_logs,
            "total_studied_time_log": t_log,
            "t_in_opt_range_log": t_in_opt_range_log,
            "battery_charge_cycles_log": battery_charge_cycles_log,
            "ere_factor_log": ere_factor_log,
            "pdm_factor_log": pdm_factor_log,
            "rated_charge_cycle_life": self.agent.battery.rated_charge_cycle_life
        }
        """

        data = Results(
            days=days,
            classification=ClassificationPerformances(
                acc=acc_log,
                prec=prec_log,
                rec=rec_log,
                f1=f1_log
            ),
            ts=ts,
            gtis_logs=gtis_logs,
            power_outputs_logs=power_outputs_logs,
            recovery_state_logs=recovery_state_logs,
            returned_by_logs=returned_by_logs,
            returned_by_none_logs=returned_by_none_logs,
            battery_logs=battery_logs,
            memory_logs=memory_logs,
            decisions_logs=decisions_logs,
            discount_logs=discount_logs,
            stage_cost_logs=stage_cost_logs,
            delegation_cost_logs=delegation_cost_logs,
            egress_rate_logs=egress_rate_logs,
            new_tasks_logs=new_tasks_logs,
            dropped_tasks_logs=dropped_tasks_logs,
            energy_del_logs=energy_del_logs,
            energy_dem_logs=energy_dem_logs,
            energy_harv_logs=energy_harv_logs,
            total_studied_time_log=t_log,
            t_in_opt_range_log=t_in_opt_range_log,
            battery_charge_cycles_log=battery_charge_cycles_log,
            ere_factor_log=ere_factor_log,
            pdm_factor_log=pdm_factor_log,
            rated_charge_cycle_life=self.agent.battery.rated_charge_cycle_life
        )

        return data
