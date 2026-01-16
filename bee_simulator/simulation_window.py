from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame
from PyQt5.QtCore import *

from simulation_stats_widget import SimulationStatsWidget
from simulation_params_widget import SimulationControlWidget
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

        """
        right_layout = QVBoxLayout(right_panel)

        upper_right = QFrame()
        upper_right.setFrameShape(QFrame.StyledPanel)
        right_layout.addWidget(upper_right, 2)  # 2/3 of right panel height

        lower_right = QFrame()
        lower_right.setFrameShape(QFrame.StyledPanel)
        right_layout.addWidget(lower_right, 1)
        """

        simulation_control_widget = SimulationControlWidget(launch_simulation_callback=self.launch_simulation,
                                                            save_plots_callback=self.save_plots,
                                                            quit_callback=self.quit_simulation)
        left_panel_layout.addWidget(simulation_control_widget, alignment=Qt.AlignHCenter | Qt.AlignVCenter)

        self.simulation_stats_widget = SimulationStatsWidget()
        # self.simulation_stats_widget.setMaximumWidth(1600)
        # layout.addWidget(self.simulation_stats_widget, alignment=Qt.AlignHCenter)
        right_panel_layout.addWidget(self.simulation_stats_widget, alignment=Qt.AlignHCenter | Qt.AlignVCenter)

        # layout.addWidget(simulation_control_widget)
        

    def launch_simulation(self):
        self.simulation.launch_simulation()
        
    def quit_simulation(self):
        self.stacked_widget.setCurrentIndex(0)

    def update_simulation_data(self, data):
        self.simulation.set_simulation_data(**data)

    def save_plots(self):
        self.simulation_stats_widget.sp.save_plots()

    def end_of_simulation(self, data):
        self.simulation_stats_widget.sp.plot(**data)
