from PyQt6 import QtWidgets, QtCore, QtGui
from devices.base import ExcitationSource


class ExcitationControlDialog(QtWidgets.QDialog):
    """Dialog for controlling excitation devices."""
    
    def __init__(self, excitation_devices, parent=None):
        super().__init__(parent)
        self.excitation_devices = excitation_devices if isinstance(excitation_devices, list) else [excitation_devices]
        self.setWindowTitle("Excitation Control")
        self.setMinimumWidth(500)
        self._build_ui()
    
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Device list
        self.device_list = QtWidgets.QListWidget()
        self.device_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(QtWidgets.QLabel("Excitation Devices:"))
        layout.addWidget(self.device_list)
        
        # Populate device list
        for device in self.excitation_devices:
            item = QtWidgets.QListWidgetItem(f"{device.name} ({type(device).__name__})")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, device)
            self.device_list.addItem(item)
        
        # Connect selection change
        self.device_list.currentRowChanged.connect(self._on_device_selected)
        
        # Control panel
        control_group = QtWidgets.QGroupBox("Channel Control")
        control_layout = QtWidgets.QVBoxLayout()
        
        # Channel selection
        channel_layout = QtWidgets.QHBoxLayout()
        channel_layout.addWidget(QtWidgets.QLabel("Channel:"))
        self.channel_combo = QtWidgets.QComboBox()
        for i in range(8):
            self.channel_combo.addItem(f"Channel {i}", i)
        channel_layout.addWidget(self.channel_combo)
        control_layout.addLayout(channel_layout)
        
        # ON/OFF buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.on_btn = QtWidgets.QPushButton("Turn ON")
        self.on_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.off_btn = QtWidgets.QPushButton("Turn OFF")
        self.off_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        button_layout.addWidget(self.on_btn)
        button_layout.addWidget(self.off_btn)
        control_layout.addLayout(button_layout)
        
        # All OFF button
        self.all_off_btn = QtWidgets.QPushButton("Turn All Channels OFF")
        self.all_off_btn.setStyleSheet("background-color: #FF9800; color: white;")
        control_layout.addWidget(self.all_off_btn)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Status display
        status_group = QtWidgets.QGroupBox("Status")
        status_layout = QtWidgets.QFormLayout()
        self.device_name_label = QtWidgets.QLabel("-")
        self.device_type_label = QtWidgets.QLabel("-")
        self.current_channel_label = QtWidgets.QLabel("-")
        self.device_state_label = QtWidgets.QLabel("-")
        self.device_state_label.setStyleSheet("font-weight: bold;")
        
        status_layout.addRow("Device:", self.device_name_label)
        status_layout.addRow("Type:", self.device_type_label)
        status_layout.addRow("Current Channel:", self.current_channel_label)
        status_layout.addRow("State:", self.device_state_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Connect buttons
        self.on_btn.clicked.connect(self._turn_on)
        self.off_btn.clicked.connect(self._turn_off)
        self.all_off_btn.clicked.connect(self._turn_all_off)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        
        # Close button
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        # Refresh button
        refresh_btn = QtWidgets.QPushButton("Refresh Status")
        refresh_btn.clicked.connect(self._refresh_status)
        layout.addWidget(refresh_btn)
        
        # Select first device
        if self.device_list.count() > 0:
            self.device_list.setCurrentRow(0)
    
    def _on_device_selected(self, row):
        """Handle device selection change."""
        if row < 0 or row >= self.device_list.count():
            return
        
        item = self.device_list.item(row)
        device = item.data(QtCore.Qt.ItemDataRole.UserRole)
        
        # Update status display
        self.device_name_label.setText(device.name)
        self.device_type_label.setText(type(device).__name__)
        
        # Update channel combo
        self.channel_combo.blockSignals(True)
        current_channel = device.get_channel() if hasattr(device, 'get_channel') else 0
        self.channel_combo.setCurrentIndex(current_channel)
        self.channel_combo.blockSignals(False)
        
        self._refresh_status()
    
    def _on_channel_changed(self, index):
        """Handle channel selection change."""
        device = self._get_current_device()
        if device and hasattr(device, 'set_channel'):
            try:
                device.set_channel(index)
                self._refresh_status()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to set channel: {e}")
    
    def _turn_on(self):
        """Turn on the current device."""
        device = self._get_current_device()
        if device and hasattr(device, 'on'):
            try:
                device.on()
                self._refresh_status()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to turn on: {e}")
    
    def _turn_off(self):
        """Turn off the current device."""
        device = self._get_current_device()
        if device and hasattr(device, 'off'):
            try:
                device.off()
                self._refresh_status()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to turn off: {e}")
    
    def _turn_all_off(self):
        """Turn off all channels on the current device."""
        device = self._get_current_device()
        if device and hasattr(device, 'all_off'):
            try:
                device.all_off()
                self._refresh_status()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to turn all off: {e}")
    
    def _refresh_status(self):
        """Refresh the status display."""
        device = self._get_current_device()
        if device:
            if hasattr(device, 'get_channel'):
                channel = device.get_channel()
                self.current_channel_label.setText(f"Channel {channel}")
            else:
                self.current_channel_label.setText("N/A")
            
            if hasattr(device, 'is_on'):
                is_on = device.is_on()
                self.device_state_label.setText("ON" if is_on else "OFF")
                self.device_state_label.setStyleSheet(
                    "color: green; font-weight: bold;" if is_on else "color: red; font-weight: bold;"
                )
            else:
                self.device_state_label.setText("Unknown")
                self.device_state_label.setStyleSheet("color: gray;")
    
    def _get_current_device(self):
        """Get the currently selected device."""
        row = self.device_list.currentRow()
        if row < 0 or row >= self.device_list.count():
            return None
        
        item = self.device_list.item(row)
        return item.data(QtCore.Qt.ItemDataRole.UserRole)