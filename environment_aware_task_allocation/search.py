from functions import *
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
import numpy as np


def grid_search(step):
    max_completed_tasks = 0

    param_grid = {'gamma_s_1': [x for x in np.arange(0, 1, step)],
                  'gamma_s_2': [x for x in np.arange(0, 1, step)],
                  'gamma_d_1': [x for x in np.arange(0, 1, step)],
                  'gamma_d_2': [x for x in np.arange(0, 1, step)]}

    opt_params = None
    
    for params in tqdm(ParameterGrid(param_grid)):
        agent = Agent(max_battery=50,
                      max_memory=50,
                      initial_battery=50,
                      initial_memory=0,
                      stage_energy_cost=0.1,
                      delegation_energy_cost=0.05,
                      gamma_s_1=params['gamma_s_1'],
                      gamma_s_2=params['gamma_s_2'],
                      gamma_d_1=params['gamma_d_1'],
                      gamma_d_2=params['gamma_d_2'],)
        for t in agent:
            continue
        num_completed_tasks_local = agent.completed_tasks_local_log
        num_completed_tasks_remote = agent.completed_tasks_remote_log
        num_completed_tasks = 0.6 * num_completed_tasks_local + 0.4 * num_completed_tasks_remote
        if num_completed_tasks > max_completed_tasks:
            max_completed_tasks = num_completed_tasks
            opt_params = params

    return opt_params
        
