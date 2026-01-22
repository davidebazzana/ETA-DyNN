from dataclasses import dataclass
import datetime
import numpy as np

@dataclass
class ClassificationPerformances:
    acc: np.array
    prec: np.array
    rec: np.array
    f1: np.array

@dataclass
class Results:
    days: list[datetime.date]
    classification: ClassificationPerformances
    ts: np.array
    gtis_logs: np.array
    power_outputs_logs: np.array
    recovery_state_logs: np.array
    returned_by_logs: dict
    returned_by_none_logs: np.array
    battery_logs: np.array
    memory_logs: np.array
    decisions_logs: np.array
    discount_logs: np.array
    stage_cost_logs: np.array
    delegation_cost_logs: np.array
    egress_rate_logs: np.array
    new_tasks_logs: np.array
    dropped_tasks_logs: np.array
    energy_del_logs: np.array
    energy_dem_logs: np.array
    energy_harv_logs: np.array
    total_studied_time_log: np.array
    t_in_opt_range_log: np.array
    battery_charge_cycles_log: np.array
    ere_factor_log: np.array
    pdm_factor_log: np.array
    rated_charge_cycle_life: int

