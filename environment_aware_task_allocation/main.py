from functions import *
from utils import *
from search import grid_search
from Q import *
import numpy as np
from scipy.stats import hypsecant
from termcolor import colored
import matplotlib.pyplot as plt
import sys
import argparse


def simulate(param_search, param_search_step, sim):
    steps = range(DAILY_STEPS)
    plt.plot(steps, [daily_solar_irradiance(t) for t in steps], '-')
    plt.xlabel("time step")
    plt.ylabel("W/m²")

    plt.show

    if param_search:
        opt_params = grid_search(param_search_step)

        if opt_params is None:
            raise RuntimeError("No parameters found")
    
        print("\n==================== RESULTS ====================")
        print("\nParameters:")
        print(f"gamma_s_1 = {opt_params['gamma_s_1']}\tgamma_s_2 = {opt_params['gamma_s_2']}")
        print(f"gamma_d_1 = {opt_params['gamma_d_1']}\tgamma_d_2 = {opt_params['gamma_d_2']}\n")
        
        agent = Agent(max_battery=50,
                      max_memory=50,
                      initial_battery=50,
                      initial_memory=0,
                      stage_energy_cost=0.1,
                      delegation_energy_cost=0.05,
                      gamma_s_1=opt_params['gamma_s_1'],
                      gamma_s_2=opt_params['gamma_s_2'],
                      gamma_d_1=opt_params['gamma_d_1'],
                      gamma_d_2=opt_params['gamma_d_2'])
    
        for t in agent:
            pass

        total_num_tasks = (agent.completed_tasks_local_log +
                           agent.completed_tasks_remote_log +
                           agent.dropped_tasks_log)
        pct_locals = (agent.completed_tasks_local_log / total_num_tasks) * 100
        pct_remote = (agent.completed_tasks_remote_log / total_num_tasks) * 100
        pct_dropped = (agent.dropped_tasks_log / total_num_tasks) * 100
        print(colored(f"{pct_locals:.2f}%", "green", attrs=["bold"]) + f" of tasks completed locally ({agent.completed_tasks_local_log})")
        print(colored(f"{pct_remote:.2f}%", "yellow", attrs=["bold"]) + f" of tasks completed remotely ({agent.completed_tasks_remote_log})")
        print(colored(f"{pct_dropped:.2f}%", "red", attrs=["bold"]) + f" of tasks dropped ({agent.dropped_tasks_log})\n")

        print("=================================================\n")

    else:
        agent = Agent(max_battery=50,
                      max_memory=50,
                      initial_battery=25,
                      initial_memory=0,
                      stage_energy_cost=0.25, # 0.1,
                      delegation_energy_cost=0.1, # 0.05,
                      gamma_s_1=0.7,
                      gamma_s_2=0.3,
                      gamma_d_1=0.9,
                      gamma_d_2=0.2)
    
        fig, ax = plt.subplots()

        t_data = []
        
        memory_line, = ax.plot(t_data, [], '-', label='memory')
        battery_line, = ax.plot(t_data, [], '-', label='battery')
        completed_tasks_local_line, = ax.plot(t_data, [], '-', label='local')
        completed_tasks_remote_line, = ax.plot(t_data, [], '-', label='remote')
        dropped_tasks_line, = ax.plot(t_data, [], '-', label='dropped')
        recovery_state_line, = ax.plot(t_data, [], '-', label='recovery')

        agent_data = {'memory': ([], memory_line),
                      'battery': ([], battery_line),
                      'completed_tasks_local_log': ([], completed_tasks_local_line),
                      'completed_tasks_remote_log': ([], completed_tasks_remote_line),
                      'dropped_tasks_log': ([], dropped_tasks_line),
                      'recovery_state_log': ([], recovery_state_line)}

        ax.legend(loc='best', frameon=False)
        
        ax.set_xlim(0, DAILY_STEPS)
        ax.set_ylim(0, 10)
        for t in agent:
            t_data.append(t)

            if sim:
                data_min = 0
                data_max = 0
            for attr, data in agent_data.items():
                data[0].append(getattr(agent, attr))

                if sim:
                    data[1].set_xdata(t_data)
                    data[1].set_ydata(data[0])
                    data_min = min(data_min, min(data[0]))
                    data_max = max(data_max, max(data[0]))
            if sim:
                ax.set_ylim(data_min, data_max)
                plt.draw()
                plt.pause(0.0001)
            if not sim:
                data_min = 0
                data_max = 0
                for attr, data in agent_data.items():
                    data[1].set_xdata(t_data)
                    data[1].set_ydata(data[0])
                    data_min = min(data_min, min(data[0]))
                    data_max = max(data_max, max(data[0]))
                ax.set_ylim(data_min, data_max)
                

        total_num_tasks = (agent.completed_tasks_local_log +
                           agent.completed_tasks_remote_log +
                           agent.dropped_tasks_log)
        pct_locals = (agent.completed_tasks_local_log / total_num_tasks) * 100
        pct_remote = (agent.completed_tasks_remote_log / total_num_tasks) * 100
        pct_dropped = (agent.dropped_tasks_log / total_num_tasks) * 100
        print(colored(f"{pct_locals:.2f}%", "green", attrs=["bold"]) + f" of tasks completed locally ({agent.completed_tasks_local_log})")
        print(colored(f"{pct_remote:.2f}%", "yellow", attrs=["bold"]) + f" of tasks completed remotely ({agent.completed_tasks_remote_log})")
        print(colored(f"{pct_dropped:.2f}%", "red", attrs=["bold"]) + f" of tasks dropped ({agent.dropped_tasks_log})\n")

        print("=================================================\n")

        if sim:
            plt.ioff()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Envirnonment-Aware Task Allocation Optimizer")
    parser.add_argument('--param-search', default=False, action=argparse.BooleanOptionalAction,
                        help='Perform parameters search')
    parser.add_argument('--param-search-step', default=0.5, type=float,
                        help='Step dimension (0, 1] when performing parameters search')
    parser.add_argument('--sim', default=False, action=argparse.BooleanOptionalAction,
                        help='Simulate real-time processing')

    args = parser.parse_args()

    simulate(args.param_search, args.param_search_step, args.sim)
