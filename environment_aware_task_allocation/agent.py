import pickle
import math
import random
import numpy as np
from scipy.stats import hypsecant
import matplotlib.pyplot as plt
from environment_aware_task_allocation.data import Dataset, Sample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from environment_aware_task_allocation.battery import Battery
from environment_aware_task_allocation.memory import Memory
from activity_emulator.solcast_dataset import SolcastDataset

class Agent():

    def __init__(self,
                 battery:Battery,
                 memory:Memory,
                 dataset:Dataset,
                 dt:float=300,
                 n_stages:int=5,
                 evidence_threshold:float=0.8,
                 future_time_window:int=5,
                 energy_threshold:float=3,
                 minimum_battery_requirement:float=5,
                 stage_energy_cost:float=0.24, # Wh
                 delegation_energy_cost:float=0.12, # Wh
                 idle_energy_cost:float=0.01, # Wh
                 gamma_s_1:float=1,
                 gamma_s_2:float=1,
                 gamma_d_1:float=1,
                 gamma_d_2:float=1):
        
        self.t_s = 0
        self.dt = dt

        self.battery = battery
        self.memory = memory
        self.dataset = dataset
        
        # Recovery state: when 1, the agent is out of energy (battery depleted) and its
        # waiting to have collect energy to restart; when 0, normal operating condition,
        # new tasks are either processed locally or delegated to the central node.
        self.recovery_state = 0

        self.MAXIMUM_SOLAR_IRRADIANCE = 1000 # W/m²
        # Maximum egress rate of bees count/minute
        self.MAXIMUM_EGRESS_RATE = 6 # 150
        self.egress_count = 0
        
        self.n_stages = n_stages
        self.evidence_threshold = evidence_threshold
        self.future_time_window = future_time_window
        self.energy_threshold = energy_threshold
        self.minimum_battery_requirement = minimum_battery_requirement
        self.maximum_future_tasks = self.future_time_window * (self.MAXIMUM_EGRESS_RATE / 60) * self.dt
        self.stage_energy_cost = stage_energy_cost
        self.delegation_energy_cost = delegation_energy_cost
        self.idle_energy_cost = idle_energy_cost

        self.gamma_s_1 = gamma_s_1
        self.gamma_s_2 = gamma_s_2
        self.gamma_d_1 = gamma_d_1
        self.gamma_d_2 = gamma_d_2

        self.recovery_state_log = [0]
        self.dropped_tasks_log = [0]
        
        self.returned_by_log = {}
        for i in range(self.n_stages + 1):
            self.returned_by_log[i] = [0]
        self.returned_by_none_log = [0]

        self.answers_log = [-1]
        self.labels_log = [-1]

        """
        self.accuracy_log = [-1]
        self.precision_log = [-1]
        self.recall_log = [-1]
        self.f1_score_log = [-1]
        """

        self.battery_log = [self.battery.soc]
        self.memory_log = [self.memory.usage_perc()]

        self.decision_log = [1]
        self.discount_val = 0
        self.discount_log = [0]
        self.stage_cost_val = 0
        self.stage_cost_log = [0]
        self.delegation_cost_val = 0
        self.delegation_cost_log = [0]

        self.egress_rate_log = [0]
        self.new_tasks_log = [0]
        self.dropped_tasks_log = [0]

        self.t = 0
        self.t_in_opt_range = 0
        self.energy_delivered_log = [0]
        self.energy_demand_log = [0]
        self.energy_harvested_log = [0]
    
    def compute(self,
                ghi:float,
                power_output:float,
                ghis_predictions:np.array,
                force_delegation:bool=False):
        decision = -1
        energy_demand = None
        if not self.memory.empty() and not self.recovery_state:
            if self.t_s == 0:
                self.sample = self.memory.dequeue()
            if force_delegation:
                decision = 0
                answer, label, returned_by, energy_used = self.apply_decision(decision)
            else:
                decision = self.decision(ghis_predictions)
                if decision == -1:
                    # Recovery state: the agent cannot perform any action because of low energy.
                    self.recovery_state = 1
                answer, label, returned_by, energy_used = self.apply_decision(decision)
        elif self.recovery_state:
            answer, label, returned_by, energy_used = -1, -1, -1, 0
        elif self.memory.empty():
            answer, label, returned_by = -1, -1, -1
            energy_used = self.battery.use(self.idle_energy_cost)
        new_tasks, egress_rate, dropped_tasks, energy_demand = self.update_state(ghi, power_output)
        
        self.t += 1
        self.update_logs(decision=decision,
                         answer=answer,
                         label=label,
                         returned_by=returned_by,
                         new_tasks=new_tasks,
                         egress_rate=egress_rate,
                         dropped_tasks=dropped_tasks,
                         power_output=power_output,
                         energy_delivered=energy_used,
                         energy_demand=energy_demand)


    def reset_logs(self):
        self.recovery_state_log = [0]
        self.returned_by_log = {}
        for i in range(self.n_stages + 1):
            self.returned_by_log[i] = [0]
        self.returned_by_none_log = [0]

        self.answers_log = [-1]
        self.labels_log = [-1]

        self.egress_rate_log = [0]
        self.new_tasks_log = [0]
        self.dropped_tasks_log = [0]

        self.battery_log = [self.battery.soc]
        self.memory_log = [self.memory.usage_perc()]
        self.decision_log = [1]
        self.discount_val = 0
        self.discount_log = [0]
        self.stage_cost_val = 0
        self.stage_cost_log = [0]
        self.delegation_cost_val = 0
        self.delegation_cost_log = [0]

        self.battery.reset_log()
        self.t = 0
        self.t_in_opt_range = 0
        self.energy_delivered_log = [0]
        self.energy_demand_log = [0]
        self.energy_harvested_log = [0]

    def reset(self):
        self.t_s = 0
        self.battery.reset()
        self.memory.reset()
        self.recovery_state = 0
        
        self.reset_logs()

    def update_logs(self,
                    decision:int,
                    answer:int,
                    label:int,
                    returned_by:int=-1,
                    new_tasks:int=0,
                    egress_rate:float=0.0,
                    dropped_tasks:int=0,
                    power_output:float=0,
                    energy_delivered:float=0,
                    energy_demand:float=0):
        """Update the performance logs. It assumes that the system has performed
        everything it had to perform in the current time interval.

        Keyword arguments:
        answer -- answer of the system to the current sample
        label -- ground truth of the current sample
        returned_by -- stage that returned the answer. If -1 nobody returned in this stage.
        """
        self.decision_log.append(decision)

        if self.recovery_state:
            self.recovery_state_log.append(self.recovery_state_log[-1] + 1)
        else:
            self.recovery_state_log.append(self.recovery_state_log[-1])

        self.answers_log.append(answer)
        self.labels_log.append(label)
        self.battery_log.append(self.battery.soc)
        self.memory_log.append(self.memory.usage_perc())

        self.discount_log.append(self.discount_val)
        self.stage_cost_log.append(self.stage_cost_val)
        self.delegation_cost_log.append(self.delegation_cost_val)

        for key in self.returned_by_log.keys():
            last_value = self.returned_by_log[key][-1]
            if key == returned_by:
                self.returned_by_log[key].append(last_value + 1)
            else:
                self.returned_by_log[key].append(last_value)
        if returned_by == -1:
            self.returned_by_none_log.append(self.returned_by_none_log[-1] + 1)
        else:
            self.returned_by_none_log.append(self.returned_by_none_log[-1])

        self.egress_rate_log.append(egress_rate) # new_tasks / (self.dt / 60))
        self.new_tasks_log.append(self.new_tasks_log[-1] + new_tasks)
        self.dropped_tasks_log.append(self.dropped_tasks_log[-1] + dropped_tasks)

        self.update_energy_metrics(power_output, energy_delivered, energy_demand)

    def update_energy_metrics(self, power_output:float=0, energy_delivered:float=0, energy_demand:float=0):
        if self.battery.soc_optimal_range[0] <= self.battery.soc <= self.battery.soc_optimal_range[1]:
            self.t_in_opt_range += 1

        self.energy_delivered_log.append(energy_delivered)
        self.energy_harvested_log.append(self.battery.compute_harvested_energy(power_output, self.dt))
        self.energy_demand_log.append(energy_demand)

    def compute_energy_demand(self, tasks):
        if isinstance(tasks, Sample):
            tasks = [tasks]

        energy_demand = 0
        for task in tasks:
            rb = task.returned_by
            if rb in range(self.n_stages):
                # One of the exits of the CNN returns the answer
                energy_demand += (rb + 1) * self.stage_energy_cost
            elif rb == self.n_stages:
                # The remote service returns the answer
                energy_demand += self.n_stages * self.stage_energy_cost + self.delegation_energy_cost
        if energy_demand == 0:
            energy_demand = self.idle_energy_cost

        return energy_demand
        
        
    def update_state(self, ghi, power_output):
        """Update the state of the system, i.e. update the battery and memory states.  If
        the memory is full, any new task is dropped. If the agent is in "recovery mode",
        new tasks are dropped.

        Keyword arguments:
        ghi -- the global horizontal
        irradiance power_output -- the power output of the solar panel

        """
        self.battery.recharge_battery(power_output, self.dt)

        # Update memory level based on solar irradiance. 
        n_new_tasks, egress_rate = self.compute_new_tasks(ghi)
        new_tasks = self.dataset.pick_random_samples(n=n_new_tasks)
        energy_demand = self.compute_energy_demand(new_tasks)
        if not self.recovery_state:
            dropped_tasks = max((self.memory.usage() + n_new_tasks) - self.memory.max_tasks, 0)

            if n_new_tasks > 0 and self.memory.usage() + n_new_tasks <= self.memory.max_tasks:
                self.memory.enqueue(new_tasks)
        else:
            dropped_tasks = n_new_tasks
            # If the battery has reached the minimum requirement, exit the "recovery mode".
            if self.battery.soc >= self.battery.discharge_cutoff_limit:
                self.recovery_state = 0

        return n_new_tasks, egress_rate, dropped_tasks, energy_demand

    def get_evidence(self):
        """Update evidence based on new computation."""
        return max(0, random.gauss(mu=float(np.tanh(self.t_s)),
                                   sigma=0.05))

    def simulate_forecast_series(self, A_series, rmse_frac=0.3197, rho=0.7):
        err_prev = 0
        forecasts = []

        for A in A_series:
            sigma = rmse_frac * A * np.sqrt(1 - rho**2)
            err = rho * err_prev + np.random.normal(0, sigma)
            forecasts.append(max(0, A + err))
            err_prev = err

        return forecasts

    def compute_new_tasks(self, ghi):
        egress_rate = (ghi / self.MAXIMUM_SOLAR_IRRADIANCE) * self.MAXIMUM_EGRESS_RATE
        self.egress_count += (egress_rate / 60) * self.dt
        if self.egress_count >= 1:
            decimal, integer = math.modf(self.egress_count)
            self.egress_count = decimal
            return int(integer), egress_rate
        else:
            return 0, egress_rate

    def compute_future_workload(self, ghi_series):
        ghi_forecasts = self.simulate_forecast_series(ghi_series)
        new_tasks = 0
        
        for ghi in ghi_forecasts:
            n_new_tasks, _ = self.compute_new_tasks(ghi)
            new_tasks += n_new_tasks
        
        return new_tasks

    def initial_task_cost(self):
        """Estimate the cost of accepting a task ("acceptance cost"). Called every time
        there is a new task to handle. It gives a measure of the current state: the
        greater the number of tasks (memory used), the greater the acceptance cost; the
        greater the energy available (battery), the smaller the acceptance cost.

        """
        # return (self.memory.usage_perc())/self.battery.soc
        # return self.memory.usage_perc()/(self.memory.usage_perc() + self.battery.soc)
        return (2 / np.pi) * np.arctan(self.memory.usage_perc()/self.battery.soc)


    def stage_cost(self, test:bool=False):
        """Estimate the cost of a stage of local elaboration in terms of the past and the
        future.

        """
        """
        c_0 = self.initial_task_cost()
        time_invested = c_0 * np.exp(self.t_s/self.n_stages)
        evidence_trend = hypsecant.pdf(self.t_s)
        return (float(self.gamma_s_1) * time_invested) - (float(self.gamma_s_2) * evidence_trend)
        """
        c_0 = self.initial_task_cost()
        # time_invested = c_0 * np.exp(self.t_s/self.n_stages)
        time_invested = c_0 * np.exp(self.t_s - self.n_stages)
        # evidence_trend = hypsecant.pdf(self.t_s)
        if self.t_s == 0:
            if test:
                evidence_trend = 0.7
            else:
                evidence_trend = self.sample.exit_0_confidence
        elif self.t_s == 1:
            if test:
                evidence_trend = 0.2
            else:
                evidence_trend = self.sample.exit_1_confidence - self.sample.exit_0_confidence
        else:
            evidence_trend = hypsecant.pdf(self.t_s)
        return (float(self.gamma_s_2) * np.exp(-evidence_trend)) + (float(self.gamma_s_1) * c_0) # time_invested)
    

    def delegation_cost(self):
        """Estimate the cost of delegating ("off-loading") the task in terms of the memory
        currently used and the time spent already to elaborate the task.

        """
        memory_occupancy = np.exp(-self.memory.usage_perc())
        # return (float(self.gamma_d_1) * (1 / self.initial_task_cost()) * memory_occupancy) + (float(self.gamma_d_2) * np.exp(self.t_s/self.n_stages))
        # return (float(self.gamma_d_1) * (1 - self.memory.usage_perc())) + (float(self.gamma_d_2) * np.exp(self.t_s/self.n_stages))
        return (float(self.gamma_d_1) * (1 - self.memory.usage_perc())) + (float(self.gamma_d_2) * np.exp(self.t_s - self.n_stages))

    # Delegation discount function. 
    def battery_discount(self, future_ghis):
        """Compute the delegation discount. f(x): [0, self.maximum_future_tasks] -> [0,
        1]. When the expected workload is locally computable, then 1 (no discount). Apply
        a discount (->0) as the expected workload increases.

        Keyword arguments:
        solar_irradiance -- the solar irradiance in the current time interval
        """
        future_workload = self.compute_future_workload(future_ghis)
        # print(f"{future_workload=}")

        # discount = 1 ---> no discount to delegation cost. discount = 0 ---> maximum
        # discount for delegation cost.
        discount = 1
        within_budget = self.battery.within_budget(future_workload * (self.stage_energy_cost))
        if within_budget:
            discount = 1
        elif not within_budget and future_workload <= self.maximum_future_tasks:
            handlable = int((self.battery.available_energy() - self.battery.cutoff_buffer) /
                            (self.stage_energy_cost))
            discount = self.maximum_future_tasks/(self.maximum_future_tasks - handlable) - (future_workload / (self.maximum_future_tasks - handlable))
        else:
            discount = 0
        
        return discount


    def memory_discount(self, future_ghis):
        future_workload = self.compute_future_workload(future_ghis)

        # discount = 1 ---> no discount to stage cost. discount = 0 ---> maximum
        # discount for stage cost.
        discount = min((self.memory.usage() + future_workload) / self.memory.max_tasks, 1)
        
        return discount

    
    def decision_cost(self, decision):
        """Compute the cost of the decision.

        Keyword arguments:
        decision -- the decision taken
        """
        if self.t_s < self.n_stages - 1:
            stage_cost = self.stage_energy_cost
        else:
            # If it is the last stage, it could be necessary to also delegate the task.
            # This could be the case if the stage does not produce an answer with enough
            # confidence.
            stage_cost = self.stage_energy_cost + self.delegation_energy_cost
        
        return stage_cost if decision else self.delegation_energy_cost
    
    def barrier_function(self, side:str):
        if side == "left":
            res = -np.log(self.battery.soc - 0.4)
        elif side == "right":
            res = -np.log(0.8 - self.battery.soc)
        else:
            raise RuntimeError("Provide left or right to the argument 'side'")
        return res

    def decision(self, ghis_prediction, test:bool=False):
        """Take a decision based on the current state and the estimates of the cost. If
        the decision is to proceed with the stage elaboration, then 1, otherwise 0
        (delegate). If the required energy to perform either actions is greater than the
        available energy, the agent should enter into the recovery state: the function
        returns -1.

        Keyword arguments:
        ghis_predictions -- list of predicted ghis
        """
        battery_discount = self.battery_discount(ghis_prediction)
        memory_discount = self.memory_discount(ghis_prediction)
        discount = battery_discount * (1 - memory_discount)
        stage_cost = self.stage_cost(test)
        delegation_cost = self.delegation_cost()
        left_barrier_function = self.barrier_function("left")
        right_barrier_function = self.barrier_function("right")
        if np.isnan(left_barrier_function):
            """If the SOC is below the lower end of its optimal range, delegate"""
            preliminary_decision = 0
        elif np.isnan(right_barrier_function):
            """If the SOC is above the upper end of its optimal range, stage locally"""
            preliminary_decision = 1
        else:
            preliminary_decision = np.where((1 - discount) * stage_cost * left_barrier_function < discount * delegation_cost * right_barrier_function, 1, 0)
        self.discount_val = discount
        self.stage_cost_val = stage_cost
        self.delegation_cost_val = delegation_cost
        # print(f"{preliminary_decision=}")
        if self.battery.within_budget(self.decision_cost(preliminary_decision)):
            # The agent can afford the cost of the action.
            return preliminary_decision
        elif self.battery.within_budget(self.decision_cost(1 - preliminary_decision)):
            # The agent cannot afford the cost of the chosen action but can afford the
            # cost of the other action, not dropping the task.
            return 1 - preliminary_decision
        else:
            # Both actions are out of battery budget, the task is dropped and the agent is
            # put in an idle state until enough energy is recovered. The event is registered.
            return -1
    
    def apply_decision(self, decision):
        """Apply the decision.
        
        Keyword arguments:
        decision -- the decision taken
        """
        returned_by = -1
        answer = self.sample.answer
        label = self.sample.label
        if decision:
            # Has elaborated locally
            # energy_used = self.battery.use(self.stage_energy_cost)
            if self.t_s == 0:
                stage_energy_cost = self.sample.preprocessing_performance + self.sample.exit_0_performance
            else:
                stage_energy_cost = self.sample.exit_1_performance
            energy_used = self.battery.use(stage_energy_cost)
            if self.sample.returned_by == self.t_s:
                returned_by = self.sample.returned_by
                answer = self.sample.answer
                label = self.sample.label
                
                self.t_s = 0
            else:
                if self.t_s < self.n_stages - 1:
                    returned_by = -1
                    answer = -1
                    label = -1
                    
                    self.t_s += 1
                else:
                    assert self.sample.returned_by == self.n_stages, f"Mismatch between the system answer and the decision.\nExpected return: {self.sample.returned_by}\nEffectively returned: {self.n_stages+1}"
                    returned_by = self.n_stages
                    answer = self.sample.answer
                    label = self.sample.label

                    # energy_used += self.battery.use(self.delegation_energy_cost)
                    energy_used += self.battery.use(self.sample.transferring_performance)

                    self.t_s = 0
        else:
            # Has delegated
            # energy_used = self.battery.use(self.delegation_energy_cost)
            energy_used = self.battery.use(self.sample.transferring_performance)

            returned_by = self.n_stages
            # TODO
            answer = self.sample.remote_answer
            label = self.sample.label

            self.t_s = 0

        energy_used = self.battery.wh_to_ah(energy_used)
        return answer, label, returned_by, energy_used

    def get_classification_performance(self):
        answers = np.array(self.answers_log)
        labels = np.array(self.labels_log)
        valid_indexes = (answers != -1) & (labels != -1)
        assert np.sum((answers != -1) ^ (labels != -1)) == 0, "Answers and labels have a different number of valid values"

        if np.sum(valid_indexes) > 0:
            answers = answers[valid_indexes]
            labels = labels[valid_indexes]
            accuracy = accuracy_score(labels, answers)
            precision = precision_score(labels, answers)
            recall = recall_score(labels, answers)
            f1 = f1_score(labels, answers)
        else:
            accuracy = np.nan
            precision = np.nan
            recall = np.nan
            f1 = np.nan
        
        return accuracy, precision, recall, f1

    def get_sustainability_performance(self):
        epsilon = 1e-8
        ere_factor = sum(self.energy_delivered_log) / (sum(self.energy_harvested_log) + epsilon)
        pdm_factor = sum(self.energy_delivered_log) / (sum(self.energy_demand_log) + epsilon)

        return self.t_in_opt_range, self.t, self.battery.n_charge_cycles, ere_factor, pdm_factor

