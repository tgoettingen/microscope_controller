#!/usr/bin/env python3
"""Simple test to verify Qt menu creation works."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PyQt6 import QtWidgets, QtCore
    print("✓ PyQt6 imported successfully")
except Exception as e:
    print(f"✗ Failed to import PyQt6: {e}")
    sys.exit(1)

try:
    app = QtWidgets.QApplication(sys.argv)
    print("✓ QApplication created")
except Exception as e:
    print(f"✗ Failed to create QApplication: {e}")
    sys.exit(1)

try:
    window = QtWidgets.QMainWindow()
    print("✓ QMainWindow created")
except Exception as e:
    print(f"✗ Failed to create QMainWindow: {e}")
    sys.exit(1)

try:
    menubar = window.menuBar()
    print("✓ Menu bar created")
except Exception as e:
    print(f"✗ Failed to create menu bar: {e}")
    sys.exit(1)

try:
    view_menu = menubar.addMenu("&View")
    print("✓ View menu created")
except Exception as e:
    print(f"✗ Failed to create View menu: {e}")
    sys.exit(1)

try:
    from PyQt6.QtGui import QAction
    test_action = QAction("Test Action", window)
    test_action.setCheckable(True)
    test_action.setChecked(False)
    view_menu.addAction(test_action)
    print("✓ Test action added to View menu")
except Exception as e:
    print(f"✗ Failed to add action to View menu: {e}")
    sys.exit(1)

print("\n✓ All Qt menu operations successful")
print("The issue might be in the main window initialization sequence")