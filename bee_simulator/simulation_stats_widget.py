from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSpinBox, QDateEdit, QFormLayout, QGridLayout, QFrame, QSizePolicy
from PyQt5.QtCore import QDate
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from simulation_plot import SimulationPlot
from time_selector import TimeSelector

class SimulationStatsWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        self.sp = SimulationPlot()
        
        canvas = FigureCanvas(self.sp.fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(canvas)

        decisions_grid_widget = QWidget()
        decisions_grid_layout = QGridLayout(decisions_grid_widget)
        decisions_grid_layout.addWidget(QLabel("Delegate:"), 0, 0)
        decisions_grid_layout.addWidget(QLabel("1252"), 0, 1)
        decisions_grid_layout.addWidget(QLabel("Local:"), 1, 0)
        decisions_grid_layout.addWidget(QLabel("3279"), 1, 1)

        sustainability_grid_widget = QWidget()
        sustainability_grid_layout = QGridLayout(sustainability_grid_widget)
        sustainability_grid_layout.addWidget(QLabel("HM:"), 1, 0)
        sustainability_grid_layout.addWidget(QLabel("0.259"), 1, 1)
        sustainability_grid_layout.addWidget(QLabel("BSI:"), 2, 0)
        sustainability_grid_layout.addWidget(QLabel("1.0"), 2, 1)
        sustainability_grid_layout.addWidget(QLabel("ERE:"), 3, 0)
        sustainability_grid_layout.addWidget(QLabel("0.201"), 3, 1)
        sustainability_grid_layout.addWidget(QLabel("PDM:"), 4, 0)
        sustainability_grid_layout.addWidget(QLabel("0.086"), 4, 1)

        performance_grid_widget = QWidget()
        performance_grid_layout = QGridLayout(performance_grid_widget)
        sustainability_header = QLabel("Sustainabilty Metrics")
        sustainability_header.setStyleSheet("font-size: 11pt; font-weight: bold;")
        performance_grid_layout.addWidget(sustainability_header, 0, 0)
        performance_grid_layout.addWidget(sustainability_grid_widget, 2, 0)
        vline = QFrame()
        vline.setFrameShape(QFrame.VLine)
        vline.setFrameShadow(QFrame.Sunken)
        performance_grid_layout.addWidget(vline, 1, 1)
        decisions_header = QLabel("Decision System Metrics")
        decisions_header.setStyleSheet("font-size: 11pt; font-weight: bold;")
        performance_grid_layout.addWidget(decisions_header, 0, 2)
        performance_grid_layout.addWidget(decisions_grid_widget, 2, 2)

        layout.addWidget(performance_grid_widget, alignment=Qt.AlignHCenter)

        time_selector = TimeSelector()
        time_selector.setMaximumWidth(300)
        layout.addWidget(time_selector, alignment=Qt.AlignHCenter)
