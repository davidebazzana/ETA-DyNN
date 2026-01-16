from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox,
    QGridLayout, QVBoxLayout
)
from PyQt5.QtCore import Qt
import sys

class TimeSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time Interval Selector")
        
        main_layout = QVBoxLayout(self)
        grid = QGridLayout()
        
        # First dropdown: interval type
        grid.addWidget(QLabel("Interval type:"), 0, 0, alignment=Qt.AlignRight)
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["All Year", "Season", "Month"])
        grid.addWidget(self.interval_combo, 0, 1)
        
        # Second dropdown: specific time choice
        grid.addWidget(QLabel("Specific:"), 1, 0, alignment=Qt.AlignRight)
        self.specific_combo = QComboBox()
        grid.addWidget(self.specific_combo, 1, 1)
        
        main_layout.addLayout(grid)
        
        # Connect signal
        self.interval_combo.currentIndexChanged.connect(self.update_specific_choices)
        
        # Initialize
        self.update_specific_choices(0)
    
    def update_specific_choices(self, index):
        """Update second dropdown depending on first dropdown selection."""
        choice = self.interval_combo.currentText()
        self.specific_combo.clear()
        
        if choice == "All Year":
            self.specific_combo.setEnabled(False)
        elif choice == "Season":
            self.specific_combo.setEnabled(True)
            self.specific_combo.addItems(["Winter", "Spring", "Summer", "Fall"])
        elif choice == "Month":
            self.specific_combo.setEnabled(True)
            self.specific_combo.addItems([
                "January", "February", "March", "April",
                "May", "June", "July", "August",
                "September", "October", "November", "December"
            ])
