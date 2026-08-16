from __future__ import annotations

from PyQt6 import QtWidgets, QtCore, QtGui
from devices.base import ExcitationSource


class ExcitationControlTab(QtWidgets.QWidget):
    """Excitation control panel as a dockable tab."""
    
    state_changed = QtCore.pyqtSignal(str, bool)  # device_name, is_on
    
    def __init__(self, excitation_devices, parent=None):
        super().__init__(parent)
        self.excitation_devices = excitation_devices if isinstance(excitation_devices, list) else [excitation_devices]
        self._build_ui()
        
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)  # Reduced spacing
        layout.setContentsMargins(4, 4, 4, 4)  # Reduced margins
        
        # Compact device and channel selector row
        selector_layout = QtWidgets.QHBoxLayout()
        selector_layout.setSpacing(4)
        
        selector_layout.addWidget(QtWidgets.QLabel("Device:"))
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.setMaximumWidth(120)
        for i, device in enumerate(self.excitation_devices):
            device_name = getattr(device, 'name', f"Device {i}")
            device_type = type(device).__name__
            self.device_combo.addItem(f"{device_name} ({device_type})", device)
        selector_layout.addWidget(self.device_combo)
        
        selector_layout.addWidget(QtWidgets.QLabel("Ch:"))
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.setMaximumWidth(80)
        for i in range(8):
            self.channel_combo.addItem(f"{i}", i)
        selector_layout.addWidget(self.channel_combo)
        
        selector_layout.addStretch()
        layout.addLayout(selector_layout)
        
        # Compact button row
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(4)
        
        self.on_btn = QtWidgets.QPushButton("ON")
        self.on_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 4px;")
        self.on_btn.setMaximumWidth(50)
        self.off_btn = QtWidgets.QPushButton("OFF")
        self.off_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 4px;")
        self.off_btn.setMaximumWidth(50)
        
        self.all_off_btn = QtWidgets.QPushButton("All OFF")
        self.all_off_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 4px;")
        self.all_off_btn.setMaximumWidth(70)
        
        button_layout.addWidget(self.on_btn)
        button_layout.addWidget(self.off_btn)
        button_layout.addWidget(self.all_off_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Compact status display (single line)
        status_layout = QtWidgets.QHBoxLayout()
        status_layout.setSpacing(8)
        
        self.device_name_label = QtWidgets.QLabel("-")
        self.device_name_label.setStyleSheet("font-size: 10px;")
        self.device_state_label = QtWidgets.QLabel("-")
        self.device_state_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        
        status_layout.addWidget(QtWidgets.QLabel("Status:"))
        status_layout.addWidget(self.device_name_label)
        status_layout.addWidget(self.device_state_label)
        status_layout.addStretch()
        
        # Small refresh button
        self.refresh_btn = QtWidgets.QPushButton("⟳")
        self.refresh_btn.setStyleSheet("padding: 2px; font-size: 10px;")
        self.refresh_btn.setMaximumWidth(25)
        self.refresh_btn.setToolTip("Refresh Status")
        status_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(status_layout)
        
        layout.addStretch(1)
        
        # Connect signals
        self.on_btn.clicked.connect(self._turn_on)
        self.off_btn.clicked.connect(self._turn_off)
        self.all_off_btn.clicked.connect(self._turn_all_off)
        self.refresh_btn.clicked.connect(self._refresh_status)
        self.device_combo.currentIndexChanged.connect(self._on_device_selected)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        
        # Select first device
        if self.device_combo.count() > 0:
            self.device_combo.setCurrentIndex(0)
    
    def _on_device_selected(self, index):
        """Handle device selection change."""
        if index < 0 or index >= self.device_combo.count():
            return
        
        device = self.device_combo.currentData()
        
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
                self.state_changed.emit(device.name, True)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to turn on: {e}")
    
    def _turn_off(self):
        """Turn off the current device."""
        device = self._get_current_device()
        if device and hasattr(device, 'off'):
            try:
                device.off()
                self._refresh_status()
                self.state_changed.emit(device.name, False)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to turn off: {e}")
    
    def _turn_all_off(self):
        """Turn off all channels on the current device."""
        device = self._get_current_device()
        if device and hasattr(device, 'all_off'):
            try:
                device.all_off()
                self._refresh_status()
                self.state_changed.emit(device.name, False)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to turn all off: {e}")
    
    def _refresh_status(self):
        """Refresh the status display."""
        device = self._get_current_device()
        if device:
            # Update device name label
            device_name = getattr(device, 'name', 'Unknown')
            if hasattr(device, 'get_channel'):
                channel = device.get_channel()
                self.device_name_label.setText(f"{device_name} (Ch {channel})")
            else:
                self.device_name_label.setText(device_name)
            
            # Update state label
            if hasattr(device, 'is_on'):
                is_on = device.is_on()
                self.device_state_label.setText("ON" if is_on else "OFF")
                self.device_state_label.setStyleSheet(
                    "color: green; font-weight: bold; font-size: 11px;" if is_on else "color: red; font-weight: bold; font-size: 11px;"
                )
            else:
                self.device_state_label.setText("?")
                self.device_state_label.setStyleSheet("color: gray; font-size: 11px;")
    
    def _get_current_device(self):
        """Get the currently selected device."""
        if self.device_combo.count() == 0:
            return None
        return self.device_combo.currentData()
    
    def set_excitation_devices(self, excitation_devices):
        """Update the excitation devices list."""
        self.excitation_devices = excitation_devices if isinstance(excitation_devices, list) else [excitation_devices]
        self.device_combo.clear()
        for i, device in enumerate(self.excitation_devices):
            device_name = getattr(device, 'name', f"Device {i}")
            device_type = type(device).__name__
            self.device_combo.addItem(f"{device_name} ({device_type})", device)
        if self.device_combo.count() > 0:
            self.device_combo.setCurrentIndex(0)
    
    def keyPressEvent(self, event):
        """Handle keyboard events - Ctrl+H hides panel, Ctrl+R reloads panel (hide and show)."""
        if event.key() == QtCore.Qt.Key.Key_H and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            # Hide panel when Ctrl+H is pressed
            parent_dock = self.parent()
            while parent_dock and not isinstance(parent_dock, QtWidgets.QDockWidget):
                parent_dock = parent_dock.parent()
            
            if parent_dock and isinstance(parent_dock, QtWidgets.QDockWidget):
                parent_dock.setVisible(False)
        elif event.key() == QtCore.Qt.Key.Key_R and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            # Reload panel (hide and show) when Ctrl+R is pressed
            parent_dock = self.parent()
            while parent_dock and not isinstance(parent_dock, QtWidgets.QDockWidget):
                parent_dock = parent_dock.parent()
            
            if parent_dock and isinstance(parent_dock, QtWidgets.QDockWidget):
                parent_dock.setVisible(False)
                parent_dock.setVisible(True)
                parent_dock.raise_()
        else:
            # Pass other key events to parent
            super().keyPressEvent(event)