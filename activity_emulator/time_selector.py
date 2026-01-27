from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox,
    QGridLayout, QVBoxLayout, QDateEdit
)
from PyQt5.QtCore import QDate
from PyQt5.QtCore import Qt
import sys

class TimeSelector(QWidget):
    def __init__(self,
                 interval_change_callback,
                 select_day_callback):
        super().__init__()
        self.setWindowTitle("Time Interval Selector")

        self.interval_change_callback = interval_change_callback
        self.select_day_callback = select_day_callback

        main_layout = QVBoxLayout(self)
        grid = QGridLayout()
        
        grid.addWidget(QLabel("Interval type:"), 0, 0, alignment=Qt.AlignRight)
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["All Year", "Season", "Month", "Day"])
        grid.addWidget(self.interval_combo, 0, 1)
        
        grid.addWidget(QLabel("Specific:"), 1, 0, alignment=Qt.AlignRight)
        self.specific_combo = QComboBox()
        grid.addWidget(self.specific_combo, 1, 1)

        grid.addWidget(QLabel("Day:"), 2, 0, alignment=Qt.AlignRight)
        self.day = QDateEdit()
        self.day.setCalendarPopup(True)
        self.day.setDate(QDate(2024, 1, 1))
        self.day.setMaximumWidth(140)
        self.day.setEnabled(False)
        grid.addWidget(self.day, 2, 1)
        
        main_layout.addLayout(grid)
        
        self.interval_combo.currentTextChanged.connect(self.update_specific_choices)
        self.specific_combo.currentTextChanged.connect(self.specific_choice)
        self.day.dateChanged.connect(self.day_choice)
        
        self.update_specific_choices(0)

    def specific_choice(self, choice):
        if choice != "":
            self.interval_change_callback(choice)

    def day_choice(self, choice):
        print(f"DAY CHOICE: {choice}")
        if choice != "":
            self.select_day_callback(choice)
        
    def update_specific_choices(self, choice):
        """Update second dropdown depending on first dropdown selection."""
        self.specific_combo.clear()
        
        if choice == "All Year":
            self.specific_combo.setEnabled(False)
            self.day.setEnabled(False)
            self.interval_change_callback(choice)
        elif choice == "Season":
            self.specific_combo.setEnabled(True)
            self.day.setEnabled(False)
            self.specific_combo.addItems(["Winter", "Spring", "Summer", "Fall"])
        elif choice == "Month":
            self.specific_combo.setEnabled(True)
            self.day.setEnabled(False)
            self.specific_combo.addItems([
                "January", "February", "March", "April",
                "May", "June", "July", "August",
                "September", "October", "November", "December"
            ])
        elif choice == "Day":
            self.day.setEnabled(True)
