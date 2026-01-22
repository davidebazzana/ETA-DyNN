import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSpinBox, QDateEdit, QFormLayout, QGridLayout, QFrame, QSizePolicy, QMessageBox
from PyQt5.QtCore import QDate
from PyQt5.QtCore import *
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from datetime import date, timedelta
import calendar

from simulation_plot import SimulationPlot
from time_selector import TimeSelector
from environment_aware_task_allocation.results import Results, ClassificationPerformances

class SimulationStatsWidget(QWidget):
    def __init__(self, save_plots_callback):
        super().__init__()

        self.save_plots_callback = save_plots_callback

        layout = QVBoxLayout(self)

        title = QLabel("Simulation Results")
        title.setAlignment(Qt.AlignLeft)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        
        layout.addStretch()

        self.sp = SimulationPlot()
        
        canvas = FigureCanvas(self.sp.fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(canvas)

        self.delegate_value = QLabel("N/A")
        self.local_value = QLabel("N/A")
        self.acc_value = QLabel("N/A")
        self.prec_value = QLabel("N/A")
        self.rec_value = QLabel("N/A")
        self.f1_value = QLabel("N/A")
        self.energy_del_value = QLabel("N/A")
        self.energy_dem_value = QLabel("N/A")
        self.energy_harv_value = QLabel("N/A")
        decisions_grid_widget = QWidget()
        decisions_grid_layout = QGridLayout(decisions_grid_widget)
        decisions_grid_layout.addWidget(QLabel("Accuracy:"), 0, 0)
        decisions_grid_layout.addWidget(self.acc_value, 0, 1)
        decisions_grid_layout.addWidget(QLabel("Precision:"), 1, 0)
        decisions_grid_layout.addWidget(self.prec_value, 1, 1)
        decisions_grid_layout.addWidget(QLabel("Recall:"), 2, 0)
        decisions_grid_layout.addWidget(self.rec_value, 2, 1)
        decisions_grid_layout.addWidget(QLabel("F1-Score:"), 3, 0)
        decisions_grid_layout.addWidget(self.f1_value, 3, 1)
        decisions_grid_layout.addWidget(QLabel("Delivered Energy:"), 4, 0)
        decisions_grid_layout.addWidget(self.energy_del_value, 4, 1)
        decisions_grid_layout.addWidget(QLabel("Energy Demand:"), 5, 0)
        decisions_grid_layout.addWidget(self.energy_dem_value, 5, 1)
        decisions_grid_layout.addWidget(QLabel("Energy Harvested:"), 6, 0)
        decisions_grid_layout.addWidget(self.energy_harv_value, 6, 1)
        decisions_grid_layout.addWidget(QLabel("Delegate:"), 7, 0)
        decisions_grid_layout.addWidget(self.delegate_value, 7, 1)
        decisions_grid_layout.addWidget(QLabel("Local:"), 8, 0)
        decisions_grid_layout.addWidget(self.local_value, 8, 1)

        self.socsi_value = QLabel("N/A")
        self.bsi_value = QLabel("N/A")
        self.ere_value = QLabel("N/A")
        self.pdm_value = QLabel("N/A")
        self.hm_value = QLabel("N/A")
        sustainability_grid_widget = QWidget()
        sustainability_grid_layout = QGridLayout(sustainability_grid_widget)
        sustainability_grid_layout.addWidget(QLabel("SOCSI:"), 0, 0)
        sustainability_grid_layout.addWidget(self.socsi_value, 0, 1)
        sustainability_grid_layout.addWidget(QLabel("BSI:"), 1, 0)
        sustainability_grid_layout.addWidget(self.bsi_value, 1, 1)
        sustainability_grid_layout.addWidget(QLabel("ERE:"), 2, 0)
        sustainability_grid_layout.addWidget(self.ere_value, 2, 1)
        sustainability_grid_layout.addWidget(QLabel("PDM:"), 3, 0)
        sustainability_grid_layout.addWidget(self.pdm_value, 3, 1)
        sustainability_grid_layout.addWidget(QLabel("HM:"), 4, 0)
        sustainability_grid_layout.addWidget(self.hm_value, 4, 1)

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

        time_selector = TimeSelector(interval_change_callback=self.interval_change,
                                     select_day_callback=self.select_day)
        time_selector.setMaximumWidth(300)
        layout.addWidget(time_selector, alignment=Qt.AlignHCenter)

        layout.addStretch()
        
        save_plots_btn = QPushButton("Save Plots")
        save_plots_btn.clicked.connect(self.save_plots)
        layout.addWidget(save_plots_btn, alignment=Qt.AlignHCenter)

    def interval_change(self, choice):
        choices = {
            "All Year": self.get_date_range(date(2024, 1, 1), date(2024, 12, 31)),
            "Winter": self.get_date_range(date(2024, 1, 1), date(2024, 3, 20)),
            "Spring": self.get_date_range(date(2024, 3, 21), date(2024, 6, 20)),
            "Summer": self.get_date_range(date(2024, 6, 21), date(2024, 9, 23)),
            "Fall": self.get_date_range(date(2024, 9, 24), date(2024, 12, 21)),
            "January": self.get_month_days(2024, 1),
            "February": self.get_month_days(2024, 2),
            "March": self.get_month_days(2024, 3),
            "April": self.get_month_days(2024, 4),
            "May": self.get_month_days(2024, 5),
            "June": self.get_month_days(2024, 6),
            "July": self.get_month_days(2024, 7),
            "August": self.get_month_days(2024, 8),
            "September": self.get_month_days(2024, 9),
            "October": self.get_month_days(2024, 10),
            "November": self.get_month_days(2024, 11),
            "December": self.get_month_days(2024, 12),
        }

        self.update_vis(choices[choice])

    def select_day(self, choice):
        self.update_vis([choice])

    def update_stats(self, simulation_results:Results):
        self.simulation_results = simulation_results
        self.update_vis()

    def update_vis(self, time_interval:list[date]|None=None):
        interval = slice(None)
        if time_interval is not None:
            interval = self.get_days_indexes(time_interval)
        
        if not isinstance(interval, slice) and len(interval) == 0:
            QMessageBox.warning(self, "No data available", "There is no data for the select time interval!", QMessageBox.Ok)
        else:
            socsi = np.sum(self.simulation_results.t_in_opt_range_log[interval]) / np.sum(self.simulation_results.total_studied_time_log[interval])
            charge_cycles_factor = np.sum(self.simulation_results.battery_charge_cycles_log[interval]) / self.simulation_results.rated_charge_cycle_life
            bsi_w_1 = 0.6
            bsi_w_2 = 0.4
            bsi = bsi_w_1 * socsi + bsi_w_2 * (1 - (charge_cycles_factor))
            ere = (1 / len(self.simulation_results.days[interval])) * np.sum(self.simulation_results.ere_factor_log[interval])
            pdm = (1 / len(self.simulation_results.days[interval])) * np.sum(self.simulation_results.pdm_factor_log[interval])
            hm = (bsi * ere * pdm) ** (1 / 3)

            self.socsi_value.setText(f"{socsi:.4f}")
            self.bsi_value.setText(f"{bsi:.4f}")
            self.ere_value.setText(f"{ere:.4f}")
            self.pdm_value.setText(f"{pdm:.4f}")
            self.hm_value.setText(f"{hm:.4f}")

            acc_mean = np.nanmean(self.simulation_results.classification.acc[interval])
            prec_mean = np.nanmean(self.simulation_results.classification.prec[interval])
            rec_mean = np.nanmean(self.simulation_results.classification.rec[interval])
            f1_mean = np.nanmean(self.simulation_results.classification.f1[interval])
            energy_del = np.sum(self.simulation_results.energy_del_logs[interval])
            energy_dem = np.sum(self.simulation_results.energy_dem_logs[interval])
            energy_harv = np.sum(self.simulation_results.energy_harv_logs[interval])
            decisions_to_delegate = np.sum(self.simulation_results.decisions_logs[interval] == 0)
            decisions_to_local = np.sum(self.simulation_results.decisions_logs[interval] == 1)

            self.acc_value.setText(f"{acc_mean:.4f}")
            self.prec_value.setText(f"{prec_mean:.4f}")
            self.rec_value.setText(f"{rec_mean:.4f}")
            self.f1_value.setText(f"{f1_mean:.4f}")
            self.energy_del_value.setText(f"{energy_del:.2f} Wh")
            self.energy_dem_value.setText(f"{energy_dem:.2f} Wh")
            self.energy_harv_value.setText(f"{energy_harv:.2f} Wh")
            self.delegate_value.setText(f"{decisions_to_delegate:.0f}")
            self.local_value.setText(f"{decisions_to_local:.0f}")

            returned_by_logs = {
                0: self.simulation_results.returned_by_logs[0][interval],
                1: self.simulation_results.returned_by_logs[1][interval],
                2: self.simulation_results.returned_by_logs[2][interval]
            }
            self.sp.plot(self.simulation_results.ts,
                         self.simulation_results.gtis_logs[interval],
                         self.simulation_results.power_outputs_logs[interval],
                         self.simulation_results.recovery_state_logs[interval],
                         returned_by_logs,
                         self.simulation_results.returned_by_none_logs[interval],
                         self.simulation_results.battery_logs[interval],
                         self.simulation_results.memory_logs[interval],
                         self.simulation_results.decisions_logs[interval],
                         self.simulation_results.discount_logs[interval],
                         self.simulation_results.stage_cost_logs[interval],
                         self.simulation_results.delegation_cost_logs[interval],
                         self.simulation_results.egress_rate_logs[interval],
                         self.simulation_results.new_tasks_logs[interval],
                         self.simulation_results.dropped_tasks_logs[interval],
                         self.simulation_results.energy_del_logs[interval],
                         self.simulation_results.energy_dem_logs[interval],
                         self.simulation_results.energy_harv_logs[interval])
        
    def save_plots(self):
        self.save_plots_callback()

    def get_date_range(self, start, end):
        return [start + timedelta(days=i) for i in range((end - start).days + 1)]

    def get_month_days(self, year, month):
        first_day = date(year, month, 1)
        _, last_day_num = calendar.monthrange(year, month)
        last_day = date(year, month, last_day_num)
        
        return self.get_date_range(first_day, last_day)

    def get_days_indexes(self, time_interval:list[date]):
        if self.simulation_results is None:
            return None
        return [i for i, d in enumerate(self.simulation_results.days) if d in time_interval]
