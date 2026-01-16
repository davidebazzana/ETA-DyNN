import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pickle
from tqdm import tqdm
from scipy.interpolate import griddata


def thresholds_plot():
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    print(f"{data=}")

    keys, inverse_indices = np.unique(data[:, :2], axis=0, return_inverse=True)
    groups = {tuple(key): data[inverse_indices == i] for i, key in enumerate(keys)}

    all_keys = list(groups.keys())
    print(f"{all_keys=}")
    for idx, group in enumerate(groups.keys()):
        print(f"{idx}: {group[0]:.2f}, {group[1]:.2f}")

    selected_keys = [all_keys[i] for i in [0, 3, 5, 10, 12, 15]]
    groups = {key: groups[key] for key in selected_keys}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))

    for i, (key, group) in enumerate(groups.items()):
        X = group[:, 2]  # Lower Threshold
        Y = group[:, 3]  # Upper Threshold
        Z = group[:, 4]  # Total energy savings %

        print(f"Key {key}: min={min(Z)}, max={max(Z)}")

        # Create a grid for surface
        xi = np.linspace(X.min(), X.max(), 30)
        yi = np.linspace(Y.min(), Y.max(), 30)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate Z onto the grid
        Zi = griddata((X, Y), Z, (Xi, Yi), method='linear')

        # Plot surface
        ax.plot_surface(Xi, Yi, Zi, color=colors[i], alpha=0.7)
    """
    for i, (key, group) in enumerate(groups.items()):
        ax.scatter(
            group[:, 2], group[:, 3], group[:, 4],
            color=colors[i],
            label=f'Group {key}',
            s=50,
            alpha=0.7
        )
    """
    """
    for group in tqdm(groups.values()):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        ax.scatter(group[:, 2], group[:, 3], group[:, 4], c='blue', marker='o', s=50, alpha=0.7)
    """
    ax.set_xlabel('Lower Threshold Exit 1')
    ax.set_ylabel('Upper Threshold Exit 1')
    ax.set_zlabel('Accuracy')
    # ax.set_title('Overall Energy Savings')

    handles = [Patch(color=colors[i], label=f'Exit 0 (lt={key[0]:.2f}, ut={key[1]:.2f})') for i, key in enumerate(groups.keys())]
    ax.legend(handles=handles)

    plt.show()


def simulation_plot():
    with open("simulation_data.pkl", "rb") as f:
        data = pickle.load(f)

    print(f"{data=}")

    keys, inverse_indices = np.unique(data[:, 0], axis=0, return_inverse=True)
    print(f"{keys=}")
    groups = {key: data[inverse_indices == i] for i, key in enumerate(keys)}

    all_keys = list(groups.keys())
    print(f"{all_keys=}")
    print(f"{groups=}")
    for idx, group in enumerate(groups.keys()):
        print(f"{idx}: {group:.2f}")

    # selected_keys = [all_keys[i] for i in [0, 3, 5, 10, 12, 15]]
    # groups = {key: groups[key] for key in selected_keys}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))

    for i, (key, group) in enumerate(groups.items()):
        X = group[:,  1]  # PV peak power
        Y = group[:,  2]  # Battery
        Z = group[:,  6]  # HM

        # Create a grid for surface
        xi = np.linspace(X.min(), X.max(), 30)
        yi = np.linspace(Y.min(), Y.max(), 30)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate Z onto the grid
        Zi = griddata((X, Y), Z, (Xi, Yi), method='linear')

        # Plot surface
        ax.plot_surface(Xi, Yi, Zi, color=colors[i], alpha=0.5) # alpha 0.7
        ax.contour(
            Xi, Yi, Zi,
            levels=[1.0],      # Z = 1 intersection
            colors='red',
            linewidths=2
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
        
    ax.set_xlabel('Battery Capacity (Ah)')
    ax.set_ylabel('Memory Capacity (# Tasks)')
    ax.set_zlabel('Battery Sustainability Index')

    handles = [Patch(color=colors[i], label=f'{key} (W)') for i, key in enumerate(groups.keys())]
    # ax.legend(handles=handles)
    ax.legend(
        handles=handles,
        title="PV Max Power Output",
        loc='lower center',
        bbox_to_anchor=(0.5, 0.8),
        ncol=3,          # adjust to taste
        frameon=True
    )

    plt.show()

    
if __name__ == "__main__":
    # thresholds_plot()
    simulation_plot()
