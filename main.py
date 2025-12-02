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

from plot_confidence import ConfidencePlot


class ConfidencePlotWidget(QWidget):
    def __init__(self,
                 train_data:np.array,
                 test_data:np.array,
                 labels:np.array,
                 title:str|None=None,
                 prev_pred:ConfidencePlot|None=None,
                 next_pred:ConfidencePlot|None=None):
        super().__init__()

        layout = QHBoxLayout(self)

        self.label = QLabel(title)
        self.label.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 16)   # font family + size
        font.setBold(True)
        font.setItalic(True)
        self.label.setFont(font)
        
        # Create the ConfidencePlot object
        self.cp = ConfidencePlot(train_data, test_data, labels, title, prev_pred, next_pred)

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

        # Instead of plt.show(), we call choose_thresholds(show=False)
        self.cp.choose_thresholds(show=False)

    def get_thresholds(self):
        try:
            t1, t2 = self.cp.get_thresholds()
            print("Selected thresholds:", t1, t2)
        except RuntimeError as e:
            print(e)


class MainWindow(QWidget):
    def __init__(self, data_path):
        super().__init__()
        self.setWindowTitle("Confidence Plot in PyQt5")
        layout = QVBoxLayout(self)
        
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        train_scores_exit_0 = data["train"]["scores"][0]["scores"]
        test_scores_exit_0 = data["test"]["scores"][0]["scores"]
        train_scores_exit_1 = data["train"]["scores"][1]["scores"]
        test_scores_exit_1 = data["test"]["scores"][1]["scores"]
        train_labels = data["train"]["labels"]
        test_labels = data["test"]["labels"]
        self.plot_widget_exit_0 = ConfidencePlotWidget(train_scores_exit_0[:, 1],
                                                       test_scores_exit_0[:, 1],
                                                       test_labels,
                                                       "Exit 0")
        self.plot_widget_exit_1 = ConfidencePlotWidget(train_scores_exit_1[:, 1],
                                                       test_scores_exit_1[:, 1],
                                                       test_labels,
                                                       "Exit 1",
                                                       self.plot_widget_exit_0.cp)
        self.plot_widget_exit_0.cp.set_next_pred(self.plot_widget_exit_1.cp)
        layout.addWidget(self.plot_widget_exit_0)
        layout.addWidget(self.plot_widget_exit_1)


def run(codename):
    app = QApplication(sys.argv)
    w = MainWindow(codename)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the data to use")
    args = parser.parse_args()

    if args.data is None:
        raise ValueError("Provide --data")
    else:
        run(args.data)
