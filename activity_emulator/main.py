from tqdm import tqdm
import argparse
import pickle
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget, QComboBox, QSizePolicy
from PyQt5.QtCore import *
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pathlib import Path
import matplotlib.pyplot as plt

from thresholds_window import ThresholdsWindow
from simulation_window import SimulationWindow


class MainWindow(QStackedWidget):
    def __init__(self, threshold_data_path, db_path, experiment_codename, dataset_codename):
        super().__init__()

        # Create pages
        self.thresholds_window = ThresholdsWindow(self, threshold_data_path, db_path, experiment_codename, dataset_codename)
        self.simulation_window = SimulationWindow(self)
        self.thresholds_window.set_simulation_widget(self.simulation_window)
        
        # Add pages to the stack
        self.addWidget(self.thresholds_window)  # index 0
        self.addWidget(self.simulation_window)  # index 1

        self.setWindowTitle("Bee Simulator")
        # self.resize(300, 150)
        self.showMaximized()

    def close_panels(self):
        self.thresholds_window.close_panels()
        
def on_exit(w:QWidget):
    w.close_panels()
    print("Gracefully stopping...")
        
def run(threshold_data_path, db_path, experiment_codename, dataset_codename):
    app = QApplication(sys.argv)
    w = MainWindow(threshold_data_path, db_path, experiment_codename, dataset_codename)
    w.show()
    app.aboutToQuit.connect(lambda: on_exit(w))
    sys.exit(app.exec_())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./scores.pkl", help="Path to the threshold")
    parser.add_argument("--db", type=str, default="./experiments.db", help="Path to the experiment db")
    parser.add_argument("--experiment", type=str, default="experiment_3", help="The codename of the experiment")
    parser.add_argument("--dataset", type=str, default="vit_validation", help="The codename of the dataset")
    args = parser.parse_args()

    if args.data is None:
        raise ValueError("Provide --data")

    if args.db is None:
        raise ValueError("Provide --db")

    if args.experiment is None:
        raise ValueError("Provide --experiment")

    if args.dataset is None:
        raise ValueError("Provide --dataset")
    
    run(args.data, args.db, args.experiment, args.dataset)
