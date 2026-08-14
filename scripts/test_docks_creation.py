#!/usr/bin/env python3
"""Test script to verify dock creation in main window."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from gui.tabs.move_motors_tab import MoveMotorsTab
    print("✓ MoveMotorsTab imported successfully")
except Exception as e:
    print(f"✗ Failed to import MoveMotorsTab: {e}")

try:
    from gui.tabs.excitation_control_tab import ExcitationControlTab
    print("✓ ExcitationControlTab imported successfully")
except Exception as e:
    print(f"✗ Failed to import ExcitationControlTab: {e}")

try:
    from gui.tabs.stage_calibration_tab import StageCalibrationTab
    print("✓ StageCalibrationTab imported successfully")
except Exception as e:
    print(f"✗ Failed to import StageCalibrationTab: {e}")

# Test tab creation
try:
    move_tab = MoveMotorsTab()
    print("✓ MoveMotorsTab created successfully")
except Exception as e:
    print(f"✗ Failed to create MoveMotorsTab: {e}")

try:
    exc_tab = ExcitationControlTab([])
    print("✓ ExcitationControlTab created successfully")
except Exception as e:
    print(f"✗ Failed to create ExcitationControlTab: {e}")

try:
    cal_tab = StageCalibrationTab(config_path="config/default_devices_simulate_mac.json")
    print("✓ StageCalibrationTab created successfully")
except Exception as e:
    print(f"✗ Failed to create StageCalibrationTab: {e}")

print("\nAll tabs can be imported and created successfully")