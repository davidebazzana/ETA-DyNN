import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pickle
from tqdm import tqdm
from scipy.interpolate import griddata
import argparse


def thresholds_plot(save:bool=False):
    with open("thresholds_search_data.pkl", "rb") as f:
        data = pickle.load(f)

    keys, inverse_indices = np.unique(data[:, :2], axis=0, return_inverse=True)
    groups = {tuple(key): data[inverse_indices == i] for i, key in enumerate(keys)}

    all_keys = list(groups.keys())
    """
    for idx, group in enumerate(groups.keys()):
        print(f"{idx}: {group[0]:.2f}, {group[1]:.2f}")
    """

    selected_keys = [all_keys[i] for i in [0, 3, 5, 10, 12, 15]]
    groups = {key: groups[key] for key in selected_keys}

    fig = plt.figure(figsize=(10, 8))
    ax_acc = fig.add_subplot(121, projection='3d')
    ax_energy = fig.add_subplot(122, projection='3d')

    colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))

    metric = {
        "ACC": {
            "ax": ax_acc,
            "data_idx": 5,
            "label": "Acc",
            "title": "Accuracy"
        },
        "ENERGY": {
            "ax": ax_energy,
            "data_idx": 4,
            "label": "%",
            "title": "Energy Savings"
        }
    }

    for v in metric.values():
        for i, (key, group) in enumerate(groups.items()):
            X = group[:,  2] 
            Y = group[:,  3]
            Z = group[:,  v["data_idx"]]

            xi = np.linspace(X.min(), X.max(), 30)
            yi = np.linspace(Y.min(), Y.max(), 30)
            Xi, Yi = np.meshgrid(xi, yi)

            Zi = griddata((X, Y), Z, (Xi, Yi), method='linear')

            v["ax"].plot_surface(Xi, Yi, Zi, color=colors[i], alpha=0.5) # alpha 0.7

        v["ax"].set_xlabel('Lower Threshold Exit 1')
        v["ax"].set_ylabel('Upper Threshold Exit 1')
        v["ax"].set_zlabel(v["label"])
        v["ax"].set_title(v["title"])

    handles = [Patch(color=colors[i], label=f'Exit 0 (lt={key[0]:.2f}, ut={key[1]:.2f})') for i, key in enumerate(groups.keys())]

    fig.legend(
        handles=handles,
        title="PV Max Power Output",
        loc='lower center',
        bbox_to_anchor=(0.5, 0.7),
        ncol=3,          # adjust to taste
        frameon=True
    )
    
    plt.show()

    if save:
        fig.savefig(
            "thresholds_results.pdf",
            bbox_inches="tight",
            pad_inches=0.5
        )


def comparison_plot(save:bool=False):
    with open("hardware_comparison_results.pkl", "rb") as f:
        data = pickle.load(f)

    values = []
    for d in data:
        socsi = np.sum(d['results'].t_in_opt_range_log) / np.sum(d['results'].total_studied_time_log)
        charge_cycles_factor = np.sum(d['results'].battery_charge_cycles_log) / d['results'].rated_charge_cycle_life
        bsi_w_1 = 0.6
        bsi_w_2 = 0.4
        bsi = bsi_w_1 * socsi + bsi_w_2 * (1 - (charge_cycles_factor))
        ere = (1 / len(d['results'].days)) * np.sum(d['results'].ere_factor_log)
        pdm = (1 / len(d['results'].days)) * np.sum(d['results'].pdm_factor_log)
        hm = (bsi * ere * pdm) ** (1 / 3)
        dropped_tasks = np.mean([dt_log[-1] for dt_log in d['results'].dropped_tasks_logs])
        # [p, b, m, bsi, ere, pdm, hm, dropped_tasks_logs[0][-1]]
        values.append([d['P_mod_STC'], d['battery'], d['memory'], bsi, ere, pdm, hm, dropped_tasks])
    values = np.array(values)
    
    keys, inverse_indices = np.unique(values[:, 0], axis=0, return_inverse=True)
    groups = {key: values[inverse_indices == i] for i, key in enumerate(keys)}

    all_keys = list(groups.keys())
    """
    for idx, group in enumerate(groups.keys()):
        print(f"{idx}: {group:.2f}")
    """
    
    fig = plt.figure(figsize=(25, 15))

    ax_hm = fig.add_subplot(151, projection='3d')
    ax_bsi = fig.add_subplot(152, projection='3d')
    ax_ere = fig.add_subplot(153, projection='3d')
    ax_pdm = fig.add_subplot(154, projection='3d')
    ax_dt = fig.add_subplot(155, projection='3d')

    colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))

    metric = {
        "BSI": {
            "ax": ax_bsi,
            "data_idx": 3,
            "label": "BSI",
            "title": "Battery Sustainability Index"
        },
        "ERE": {
            "ax": ax_ere,
            "data_idx": 4,
            "label": "ERE",
            "title": "Energy Efficiency Reliability"
        },
        "PDM": {
            "ax": ax_pdm,
            "data_idx": 5,
            "label": "PDM",
            "title": "Power Demand Match"
        },
        "HM": {
            "ax": ax_hm,
            "data_idx": 6,
            "label": "HM",
            "title": "Holistic Metric"
        },
        "DT": {
            "ax": ax_dt,
            "data_idx": 7,
            "label": "Count",
            "title": "Dropped Tasks"
        }
    }
    
    for v in metric.values():
        for i, (key, group) in enumerate(groups.items()):
            X = group[:,  1]  # Battery
            Y = group[:,  2]  # Memory
            Z = group[:,  v["data_idx"]]

            # Create a grid for surface
            xi = np.linspace(X.min(), X.max(), 30)
            yi = np.linspace(Y.min(), Y.max(), 30)
            Xi, Yi = np.meshgrid(xi, yi)
        
            # Interpolate Z onto the grid
            Zi = griddata((X, Y), Z, (Xi, Yi), method='linear')

            # Plot surface
            v["ax"].plot_surface(Xi, Yi, Zi, color=colors[i], alpha=0.5) # alpha 0.7
            v["ax"].contour(
                Xi, Yi, Zi,
                levels=[1.0],      # Z = 1 intersection
                colors='red',
                linewidths=2
            )

        v["ax"].set_xlabel('Battery Capacity (Ah)')
        v["ax"].set_ylabel('Memory Capacity (# Tasks)')
        v["ax"].set_zlabel(v["label"])
        v["ax"].set_title(v["title"])

    handles = [Patch(color=colors[i], label=f'{key} (W)') for i, key in enumerate(groups.keys())]
    fig.legend(
        handles=handles,
        title="PV Max Power Output",
        loc='lower center',
        bbox_to_anchor=(0.5, 0.7),
        ncol=3,          # adjust to taste
        frameon=True
    )

    plot_plane_1 = False
    if plot_plane_1:
        x_plane = np.linspace(X.min(), X.max(), 30)
        y_plane = np.linspace(Y.min(), Y.max(), 30)
        X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
        Z_plane = np.ones_like(X_plane) * 1.0  # height = 1

        ax.plot_surface(
            X_plane,
            Y_plane,
            Z_plane,
            color='red',
            alpha=0.5,
            edgecolor='none'
        )

    plt.show()

    if save:
        fig.savefig(
            "hardware_dim_sustainability_metrics.pdf",
            bbox_inches="tight",
            pad_inches=0.5
        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Plotting Utility")
    parser.add_argument('--save', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--thresholds', action=argparse.BooleanOptionalAction)
    parser.add_argument('--hardware-comparison', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    if args.thresholds:
        thresholds_plot(save=args.save)
    if args.hardware_comparison:
        comparison_plot(save=args.save)
