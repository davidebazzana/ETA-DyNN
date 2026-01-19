from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSpinBox, QDateEdit, QFormLayout, QGridLayout, QFrame, QCheckBox, QSpacerItem, QSizePolicy
from PyQt5.QtCore import QDate
from PyQt5.QtCore import *
from PyQt5.QtGui import QFont


class SimulationControlWidget(QWidget):

    def __init__(self,
                 launch_simulation_callback,
                 save_simulation_callback,
                 load_simulation_callback,
                 compare_hardware_callback,
                 quit_callback):
        super().__init__()

        self.launch_simulation_callback = launch_simulation_callback
        self.load_simulation_callback = load_simulation_callback
        self.save_simulation_callback = save_simulation_callback
        self.compare_hardware_callback = compare_hardware_callback
        self.quit_callback = quit_callback

        layout = QVBoxLayout(self)

        title = QLabel("Simulation Parameters")
        title.setAlignment(Qt.AlignLeft)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        
        layout.addStretch()

        sim_params_form_layout = QFormLayout()

        form_labels_font = QFont()
        form_labels_font.setBold(True)

        hardware_label = QLabel("Hardware Dimensioning")
        hardware_label.setFont(form_labels_font)

        self.pv_input = QSpinBox()
        self.pv_input.setMaximum(10**9)
        self.pv_input.setValue(80)
        self.pv_input.setMaximumWidth(140)

        self.battery_initial_soc_input = QSpinBox()
        self.battery_initial_soc_input.setMaximum(100)
        self.battery_initial_soc_input.setValue(50)
        self.battery_initial_soc_input.setMaximumWidth(140)

        self.battery_capacity_input = QSpinBox()
        self.battery_capacity_input.setMaximum(10**9)
        self.battery_capacity_input.setValue(200)
        self.battery_capacity_input.setMaximumWidth(140)

        self.memory_capacity_input = QSpinBox()
        self.memory_capacity_input.setMaximum(10**9)
        self.memory_capacity_input.setValue(2000)
        self.memory_capacity_input.setMaximumWidth(140)

        time_label = QLabel("Time Interval")
        time_label.setFont(form_labels_font)
        
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate(2024, 1, 1))
        self.start_date.setMaximumWidth(140)

        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate(2024, 12, 31))
        self.end_date.setMaximumWidth(140)

        spacer = QSpacerItem(
            0, 15,
            QSizePolicy.Minimum,
            QSizePolicy.Fixed
        )

        system_label = QLabel("System")
        system_label.setFont(form_labels_font)

        self.force_delegation_checkbox = QCheckBox()
        self.force_delegation_checkbox.setChecked(False)

        self.daily_reset_checkbox = QCheckBox()
        self.daily_reset_checkbox.setChecked(False)

        sim_params_form_layout.addRow(hardware_label)
        sim_params_form_layout.addRow("PV Max Power Output at STC:", self.pv_input)
        sim_params_form_layout.addRow("Battery initial SOC:", self.battery_initial_soc_input)
        sim_params_form_layout.addRow("Battery capacity:", self.battery_capacity_input)
        sim_params_form_layout.addRow("Memory capacity:", self.memory_capacity_input)
        sim_params_form_layout.addItem(spacer)
        sim_params_form_layout.addRow(time_label)
        sim_params_form_layout.addRow("Start date:", self.start_date)
        sim_params_form_layout.addRow("End date:", self.end_date)
        sim_params_form_layout.addItem(spacer)
        sim_params_form_layout.addRow(system_label)
        sim_params_form_layout.addRow("Force delegation:", self.force_delegation_checkbox)
        sim_params_form_layout.setAlignment(self.force_delegation_checkbox, Qt.AlignRight)
        sim_params_form_layout.addRow("Daily reset the agent:", self.daily_reset_checkbox)
        sim_params_form_layout.setAlignment(self.daily_reset_checkbox, Qt.AlignRight)

        form_widget = QWidget()
        form_widget.setLayout(sim_params_form_layout)
        form_widget.setMaximumWidth(300)
        layout.addWidget(form_widget, alignment=Qt.AlignHCenter)

        layout.addStretch()
        
        launch_simulation_btn = QPushButton("Launch Simulation")
        launch_simulation_btn.clicked.connect(self.launch_simulation)
        
        compare_hardware_btn = QPushButton("Compare Hardware")
        compare_hardware_btn.clicked.connect(self.compare_hardware)
        
        save_simulation_btn = QPushButton("Save Simulation")
        save_simulation_btn.clicked.connect(self.save_simulation)

        load_simulation_btn = QPushButton("Load Simulation")
        load_simulation_btn.clicked.connect(self.load_simulation)

        quit_btn = QPushButton("Quit Simulation")
        quit_btn.clicked.connect(self.quit)

        btn_width = 200
        launch_simulation_btn.setMaximumWidth(form_widget.maximumWidth())
        launch_simulation_btn.setMinimumWidth(btn_width)
        compare_hardware_btn.setMaximumWidth(form_widget.maximumWidth())
        compare_hardware_btn.setMinimumWidth(btn_width)
        save_simulation_btn.setMaximumWidth(form_widget.maximumWidth())
        save_simulation_btn.setMinimumWidth(btn_width)
        load_simulation_btn.setMaximumWidth(form_widget.maximumWidth())
        load_simulation_btn.setMinimumWidth(btn_width)
        quit_btn.setMaximumWidth(form_widget.maximumWidth())
        quit_btn.setMinimumWidth(btn_width)
        
        layout.addWidget(launch_simulation_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(compare_hardware_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(save_simulation_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(load_simulation_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(quit_btn, alignment=Qt.AlignHCenter)

    def launch_simulation(self):
        self.launch_simulation_callback({
            "P_mod_STC": self.pv_input.value(),
            "battery_initial_soc": self.battery_initial_soc_input.value(),
            "battery_capacity": self.battery_capacity_input.value(),
            "memory_capacity": self.memory_capacity_input.value(),
            "start_date": self.start_date.date().toPyDate(),
            "end_date": self.end_date.date().toPyDate(),
            "force_delegation": self.force_delegation_checkbox.isChecked(),
            "daily_reset": self.daily_reset_checkbox.isChecked()
        })

    def compare_hardware(self):
        self.compare_hardware_callback()

    def quit(self):
        self.quit_callback()

    def save_simulation(self):
        self.save_simulation_callback()

    def load_simulation(self):
        self.load_simulation_callback()
