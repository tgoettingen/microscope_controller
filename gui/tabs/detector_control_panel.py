"""
Minimal Detector Control Panel - Provides detector control widgets in a compact layout.

This is a simplified version that only provides the control layout,
relying on DetectorImagePanel for advanced features like tooltips and overlays.
"""

from PyQt6 import QtWidgets, QtCore, QtGui


class DetectorControlPanel(QtWidgets.QWidget):
    """Holds detector visibility/stream controls in a vertical layout."""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Keep this panel compact even when placed in a tall dock.
        # The content can scroll instead of forcing the dock to grow.
        self._max_panel_height = 200
        self.group = QtWidgets.QGroupBox("Detectors")
        self.group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; }")
        # Foldable/collapsible panel
        try:
            self.group.setCheckable(True)
            self.group.setChecked(True)
        except Exception:
            pass

        # Put controls inside an inner widget so we can hide/show them
        self._content = QtWidgets.QWidget()
        self.vlayout = QtWidgets.QVBoxLayout(self._content)
        self.vlayout.setContentsMargins(2, 2, 2, 2)
        try:
            self.vlayout.setSpacing(1)
        except Exception:
            pass

        # Make the control list scroll when there are many detectors.
        self._scroll = QtWidgets.QScrollArea(self.group)
        self._scroll.setWidgetResizable(True)
        try:
            self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        except Exception:
            pass
        self._scroll.setWidget(self._content)

        g_layout = QtWidgets.QVBoxLayout(self.group)
        g_layout.setContentsMargins(4, 8, 4, 4)
        g_layout.setSpacing(2)
        g_layout.addWidget(self._scroll)

        try:
            self.group.toggled.connect(self._scroll.setVisible)
        except Exception:
            pass

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.group)

        # Size policy: prefer not to expand vertically.
        try:
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
            self.group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
            self._scroll.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
            self.setMaximumHeight(self._max_panel_height)
            self.group.setMaximumHeight(self._max_panel_height)
            self._scroll.setMaximumHeight(self._max_panel_height)
        except Exception:
            pass

    def add_control_row(self, widget):
        self.vlayout.addWidget(widget)

        # Keep compact height as controls are added dynamically.
        try:
            self.adjustSize()
        except Exception:
            pass
    
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
