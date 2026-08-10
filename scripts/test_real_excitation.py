#!/usr/bin/env python3
"""Test script for real ExcitationDevice using SerialLink protocol."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devices.excitation_device import ExcitationDevice


def test_real_excitation_device():
    """Test the real excitation device with SerialLink."""
    print("Testing Real ExcitationDevice with SerialLink...")
    
    # Create real device with the specified port
    device = ExcitationDevice(
        name="excitation",
        port="/dev/cu.usbmodemM43210051",
        channel=0,
        simulate=False
    )
    
    try:
        # Test basic functionality
        device.connect()
        print("✓ Real device connected to /dev/cu.usbmodemM43210051")
        
        # Test capabilities
        caps = device.get_capabilities()
        print(f"✓ Device capabilities: {caps}")
        
        # Test channel control
        device.set_channel(0)
        print(f"✓ Channel set to {device.get_channel()}")
        
        # Test on/off
        device.on()
        print(f"✓ Device turned ON (is_on: {device.is_on()})")
        
        # Wait a bit
        import time
        time.sleep(1)
        
        device.off()
        print(f"✓ Device turned OFF (is_on: {device.is_on()})")
        
        # Test channel switching
        device.set_channel(1)
        print(f"✓ Channel changed to {device.get_channel()}")
        
        device.on()
        print(f"✓ Device turned ON on channel 1 (is_on: {device.is_on()})")
        
        time.sleep(1)
        
        device.off()
        print(f"✓ Device turned OFF on channel 1 (is_on: {device.is_on()})")
        
        # Test all off
        device.on()
        device.all_off()
        print(f"✓ All channels turned OFF (is_on: {device.is_on()})")
        
        device.disconnect()
        print("✓ Real device disconnected")
        
        return True
    except Exception as e:
        print(f"✗ Real device test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Real Excitation Device Test")
    print("=" * 60)
    
    result = test_real_excitation_device()
    
    print("\n" + "=" * 60)
    if result:
        print("✓ Real device test passed")
        sys.exit(0)
    else:
        print("✗ Real device test failed")
        sys.exit(1)