#!/usr/bin/env python3
"""Simple test for ExcitationDevice with SerialLink."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devices.excitation_device import ExcitationDevice
import time

def test_simple():
    """Simple connection and ON/OFF test."""
    print("Simple ExcitationDevice test...")
    
    device = ExcitationDevice(
        name="TestExcitation",
        port="/dev/cu.usbmodemM43210051",
        channel=0,
        simulate=False
    )
    
    try:
        print("Connecting...")
        device.connect()
        print(f"✓ Connected to {device._port}")
        
        print("Turning ON...")
        device.on()
        print(f"✓ Device ON (is_on: {device.is_on()})")
        
        time.sleep(2)
        
        print("Turning OFF...")
        device.off()
        print(f"✓ Device OFF (is_on: {device.is_on()})")
        
        print("Disconnecting...")
        device.disconnect()
        print("✓ Disconnected")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_simple()
    sys.exit(0 if result else 1)