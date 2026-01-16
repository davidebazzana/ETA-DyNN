from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSpinBox, QDateEdit, QFormLayout, QGridLayout, QFrame
from PyQt5.QtCore import QDate
from PyQt5.QtCore import *


class SimulationControlWidget(QWidget):

    def __init__(self,
                 launch_simulation_callback,
                 save_plots_callback,
                 quit_callback):
        super().__init__()

        self.launch_simulation_callback = launch_simulation_callback
        self.save_plots_callback = save_plots_callback
        self.quit_callback = quit_callback

        layout = QVBoxLayout(self)

        sim_params_form_layout = QFormLayout()
        pv_input = QSpinBox()
        pv_input.setMaximum(10**9)
        pv_input.setValue(80)
        pv_input.setMaximumWidth(140)

        battery_input = QSpinBox()
        battery_input.setMaximum(10**9)
        battery_input.setValue(200)
        battery_input.setMaximumWidth(140)

        memory_input = QSpinBox()
        memory_input.setMaximum(10**9)
        memory_input.setValue(2000)
        memory_input.setMaximumWidth(140)

        start_date = QDateEdit()
        start_date.setCalendarPopup(True)
        start_date.setDate(QDate(2024, 1, 1))
        start_date.setMaximumWidth(140)

        end_date = QDateEdit()
        end_date.setCalendarPopup(True)
        end_date.setDate(QDate(2024, 12, 31))
        end_date.setMaximumWidth(140)

        sim_params_form_layout.addRow("PV Max Power Output at STC:", pv_input)
        sim_params_form_layout.addRow("Battery capacity:", battery_input)
        sim_params_form_layout.addRow("Memory capacity:", memory_input)
        sim_params_form_layout.addRow("Start date:", start_date)
        sim_params_form_layout.addRow("End date:", end_date)

        form_widget = QWidget()
        form_widget.setLayout(sim_params_form_layout)
        form_widget.setMaximumWidth(300)
        layout.addWidget(form_widget, alignment=Qt.AlignHCenter)
        
        launch_simulation_btn = QPushButton("Launch simulation")
        launch_simulation_btn.clicked.connect(self.launch_simulation)
        
        save_plots_btn = QPushButton("Save Plots")
        save_plots_btn.clicked.connect(self.save_plots)
        
        save_simulation_btn = QPushButton("Save Simulation")
        save_simulation_btn.clicked.connect(self.save_simulation)

        load_simulation_btn = QPushButton("Load Simulation")
        load_simulation_btn.clicked.connect(self.load_simulation)

        quit_btn = QPushButton("Quit Simulation")
        quit_btn.clicked.connect(self.quit)

        btn_width = 200
        launch_simulation_btn.setMaximumWidth(form_widget.maximumWidth())
        launch_simulation_btn.setMinimumWidth(btn_width)
        save_plots_btn.setMaximumWidth(form_widget.maximumWidth())
        save_plots_btn.setMinimumWidth(btn_width)
        save_simulation_btn.setMaximumWidth(form_widget.maximumWidth())
        save_simulation_btn.setMinimumWidth(btn_width)
        load_simulation_btn.setMaximumWidth(form_widget.maximumWidth())
        load_simulation_btn.setMinimumWidth(btn_width)
        quit_btn.setMaximumWidth(form_widget.maximumWidth())
        quit_btn.setMinimumWidth(btn_width)

        layout.addWidget(launch_simulation_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(save_plots_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(save_simulation_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(load_simulation_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(quit_btn, alignment=Qt.AlignHCenter)

    def launch_simulation(self):
        self.launch_simulation_callback()

    def quit(self):
        self.quit_callback()

    def save_plots(self):
        self.save_plots_callback()

    def save_simulation(self):
        pass

    def load_simulation(self):
        pass
