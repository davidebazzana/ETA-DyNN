import argparse
import pickle
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import *
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pathlib import Path
import numpy as np

from confidence_plot import ConfidencePlot
from summary_plot import SummaryPlot


class SummaryPlotWidget(QWidget):
    def __init__(self,
                 db_path:str,
                 confidence_plot_exit_0:ConfidencePlot,
                 confidence_plot_exit_1:ConfidencePlot,
                 experiment_codename:str,
                 dataset_codename:str):
        super().__init__()

        layout = QHBoxLayout(self)

        title = f"Global Results"
        self.label = QLabel(title)
        self.label.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 16)   # font family + size
        font.setBold(True)
        font.setItalic(True)
        self.label.setFont(font)

        self.sp = SummaryPlot(db_path, confidence_plot_exit_0, confidence_plot_exit_1,
                              experiment_codename, dataset_codename)

        # Create a Matplotlib canvas widget and insert the figure
        self.canvas = FigureCanvas(self.sp.fig)

        layout.addWidget(self.label)
        layout.addWidget(self.canvas)

class ConfidencePlotWidget(QWidget):
    def __init__(self,
                 train_data:np.array,
                 train_labels:np.array,
                 db_path:str,
                 experiment_codename:str,
                 dataset_codename:str,
                 model_codename:str,
                 exit_idx:int|None=None,
                 prev_pred:ConfidencePlot|None=None,
                 next_pred:ConfidencePlot|None=None):
        super().__init__()

        layout = QHBoxLayout(self)

        title = f"Model: {model_codename}"
        if exit_idx is not None: title += f"\nExit: {exit_idx}"
        self.label = QLabel(title)
        self.label.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 16)   # font family + size
        font.setBold(True)
        font.setItalic(True)
        self.label.setFont(font)
        
        # Create the ConfidencePlot object
        self.cp = ConfidencePlot(train_data, train_labels, db_path, experiment_codename,
                                 dataset_codename, model_codename, exit_idx, prev_pred, next_pred)

        # Create a Matplotlib canvas widget and insert the figure
        self.canvas = FigureCanvas(self.cp.fig)

        layout.addWidget(self.label)
        layout.addWidget(self.canvas)

        # Add a button to read thresholds when Enter would normally close the plot
        """
        self.btn = QPushButton("Get Selected Thresholds")
        self.btn.clicked.connect(self.get_thresholds)
        layout.addWidget(self.btn)
        """

    def get_thresholds(self):
        try:
            t1, t2 = self.cp.get_thresholds()
            print("Selected thresholds:", t1, t2)
        except RuntimeError as e:
            print(e)


class MainWindow(QWidget):
    def __init__(self, threshold_data_path, db_path,
                 experiment_codename, dataset_codename):
        super().__init__()
        self.setWindowTitle("Confidence Plot in PyQt5")
        layout = QVBoxLayout(self)
        
        with open(threshold_data_path, "rb") as f:
            data = pickle.load(f)
        train_scores_exit_0 = data["train"]["scores"][0]["scores"]
        train_scores_exit_1 = data["train"]["scores"][1]["scores"]
        train_labels = data["train"]["labels"]
        self.plot_widget_exit_0 = ConfidencePlotWidget(train_scores_exit_0[:, 1],
                                                       train_labels,
                                                       db_path,
                                                       experiment_codename,
                                                       dataset_codename,
                                                       "ee_cnn",
                                                       0)
        self.plot_widget_exit_1 = ConfidencePlotWidget(train_scores_exit_1[:, 1],
                                                       train_labels,
                                                       db_path,
                                                       experiment_codename,
                                                       dataset_codename,
                                                       "ee_cnn",
                                                       1,
                                                       self.plot_widget_exit_0.cp)
        self.plot_widget_exit_0.cp.set_next_pred(self.plot_widget_exit_1.cp)
        layout.addWidget(self.plot_widget_exit_0)
        layout.addWidget(self.plot_widget_exit_1)

        self.confidence_panels = [self.plot_widget_exit_0.cp,
                                  self.plot_widget_exit_1.cp]

        self.plot_widget_summary = SummaryPlotWidget(db_path,
                                                     self.confidence_panels[0],
                                                     self.confidence_panels[1],
                                                     experiment_codename,
                                                     dataset_codename)

        layout.addWidget(self.plot_widget_summary)
                         
        self.resize_histograms()
        
    def resize_histograms(self):
        max_y = max([cp.max_hist_y for cp in self.confidence_panels])
        new_y_lim = max_y + (0.05 * max_y)
        for cp in self.confidence_panels: cp.set_hist_ylim(0, new_y_lim)

    def close_panels(self):
        for cp in self.confidence_panels:
            cp.close()
        self.plot_widget_summary.sp.close()

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
    parser.add_argument("--data", type=str, help="Path to the threshold")
    parser.add_argument("--db", type=str, help="Path to the experiment db")
    parser.add_argument("--experiment", type=str, help="The codename of the experiment")
    parser.add_argument("--dataset", type=str, help="The codename of the dataset")
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
