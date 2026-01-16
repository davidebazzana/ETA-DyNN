import pickle
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import *
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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

    
class ThresholdsWindow(QWidget):
    def __init__(self, stacked_widget, threshold_data_path, db_path,
                 experiment_codename, dataset_codename):
        super().__init__()

        self.stacked_widget = stacked_widget

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

        self.set_thresholds_btn = QPushButton("Set thresholds (lt_0=0.1, ut_0=0.9), (lt_1=0.2, ut_1=0.8)")
        self.set_thresholds_btn.clicked.connect(self.set_thresholds)
        layout.addWidget(self.set_thresholds_btn)

        self.compare_thresholds_btn = QPushButton("Compare thresholds performance")
        self.compare_thresholds_btn.clicked.connect(self.compare_thresholds)
        layout.addWidget(self.compare_thresholds_btn)

        self.btn = QPushButton("Proceed to simulation")
        self.btn.clicked.connect(self.switch_to_simulation)
        layout.addWidget(self.btn)
                         
        self.resize_histograms()
        
    def resize_histograms(self):
        max_y = max([cp.max_hist_y for cp in self.confidence_panels])
        new_y_lim = max_y + (0.05 * max_y)
        for cp in self.confidence_panels: cp.set_hist_ylim(0, new_y_lim)

    def set_simulation_widget(self, simulation_widget):
        self.simulation_widget = simulation_widget

    def get_classification_data(self):
        data = {
            'returned_by_e0': np.copy(self.plot_widget_summary.sp.returned_by_e0),
            'returned_by_e1': np.copy(self.plot_widget_summary.sp.returned_by_e1),
            'returned_by_vit4v': np.copy(self.plot_widget_summary.sp.returned_by_vit4v),
            'global_answers': np.copy(self.plot_widget_summary.sp.global_answers),
            'vit4v_answers': np.copy(self.plot_widget_summary.sp.vit4v_answers),
            'labels': np.copy(self.plot_widget_summary.sp.labels)
        }

        return data
        
    def switch_to_simulation(self):
        self.simulation_widget.update_simulation_data(self.get_classification_data())
        self.stacked_widget.setCurrentIndex(1)

    def set_thresholds(self):
        self.plot_widget_exit_0.cp.set_thresholds(0.1, 0.9) # (0.1, 0.9)
        self.plot_widget_exit_1.cp.set_thresholds(0.2, 0.8) # (0.2, 0.8)

    def compare_thresholds(self):
        n = 4
        lower_0 = np.linspace(0.01, 0.20, n)
        upper_0 = np.linspace(0.80, 0.99, n)
        lower_1 = np.linspace(0.01, 0.30, n)
        upper_1 = np.linspace(0.70, 0.99, n)

        data = []
        for l0 in tqdm(lower_0):
            for u0 in upper_0:
                for l1 in lower_1:
                    for u1 in upper_1:
                        self.plot_widget_exit_0.cp.set_thresholds(l0, u0)
                        self.plot_widget_exit_1.cp.set_thresholds(l1, u1)
                        saving = ((self.plot_widget_summary.sp.costs["tot_energy"]["baseline"]["total"] - self.plot_widget_summary.sp.costs["tot_energy"]["system"]["total"])/self.plot_widget_summary.sp.costs["tot_energy"]["baseline"]["total"])*100
                        acc = self.plot_widget_summary.sp.metrics["acc"]
                        data.append(np.array([l0, u0, l1, u1, saving, acc]))
        data = np.array(data)

        with open("thresholds_search_data.pkl", "wb") as f:
            pickle.dump(data, f)

    def close_panels(self):
        for cp in self.confidence_panels:
            cp.close()
        self.plot_widget_summary.sp.close()
