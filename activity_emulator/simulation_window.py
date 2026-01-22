import pickle
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QAction, QFileDialog, QMessageBox
from PyQt5.QtCore import *

from simulation_stats_widget import SimulationStatsWidget
from simulation_control_widget import SimulationControlWidget
from simulation import Simulation


class SimulationWindow(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        self.simulation = Simulation(self.end_of_simulation)

        main_layout = QHBoxLayout(self)

        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel_layout = QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel, 1)  # stretch factor = 1 (1/3)

        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, 4)  # stretch factor = 2 (2/3)

        simulation_control_widget = SimulationControlWidget(launch_simulation_callback=self.launch_simulation,
                                                            save_simulation_callback=self.save_simulation,
                                                            load_simulation_callback=self.load_simulation,
                                                            compare_hardware_callback=self.compare_hardware,
                                                            quit_callback=self.quit_simulation)
        left_panel_layout.addWidget(simulation_control_widget, alignment=Qt.AlignHCenter)

        self.simulation_stats_widget = SimulationStatsWidget(save_plots_callback=self.save_plots)
        # self.simulation_stats_widget.setMaximumWidth(1600)
        # layout.addWidget(self.simulation_stats_widget, alignment=Qt.AlignHCenter)
        right_panel_layout.addWidget(self.simulation_stats_widget, alignment=Qt.AlignHCenter)

        self.simulation_data_results = None
        # layout.addWidget(simulation_control_widget)
        
    def compare_hardware(self):
        self.simulation.compare_hardware()

    def launch_simulation(self, simulation_params):
        self.simulation.launch_simulation(**simulation_params,
                                          end_callback=self.end_of_simulation,
                                          solcast_partition="validation")
        
    def quit_simulation(self):
        self.stacked_widget.setCurrentIndex(0)

    def update_simulation_data(self, data):
        self.simulation.set_simulation_data(**data)

    def save_plots(self):
        self.simulation_stats_widget.sp.save_plots()

    def end_of_simulation(self, data):
        self.simulation_data_results = data
        self.simulation_stats_widget.update_stats(self.simulation_data_results)

    def load_simulation(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Simulation Results",
            "",
            "Pickle Files (*.pkl);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "rb") as f:
                self.simulation_data_results = pickle.load(f)
            print("Loaded:", path)
            self.simulation_stats_widget.update_stats(self.simulation_data_results)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def save_simulation(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Simulation Results",
            "",
            "Pickle Files (*.pkl);;All Files (*)"
        )
        if not path:
            return

        if not path.endswith(".pkl"):
            path += ".pkl"

        try:
            with open(path, "wb") as f:
                pickle.dump(self.simulation_data_results, f)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
