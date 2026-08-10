#!/usr/bin/env python3
"""Test script for ExcitationDevice and ExcitationAxis functionality."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devices.excitation_device import ExcitationDevice, SimulatedExcitationDevice
from devices.base import ExcitationSource
from core.multiaxis import ExcitationAxis, AxisConfig


def test_simulated_excitation_device():
    """Test the simulated excitation device."""
    print("Testing SimulatedExcitationDevice...")
    
    # Create simulated device
    # device = SimulatedExcitationDevice("TestExcitation")
    device = ExcitationDevice("TestExcitation",port = '/dev/cu.usbmodemM43210051',simulate = False)
    # Test basic functionality
    try:
        device.connect()
        print("✓ Simulated device connected")
        
        # Test capabilities
        caps = device.get_capabilities()
        print(f"✓ Device capabilities: {caps}")
        
        # Test channel control
        device.set_channel(0)
        print(f"✓ Channel set to {device.get_channel()}")
        
        device.set_channel(2)
        print(f"✓ Channel changed to {device.get_channel()}")
        
        # Test on/off
        device.on()
        print(f"✓ Device turned ON (is_on: {device.is_on()})")
        
        device.off()
        print(f"✓ Device turned OFF (is_on: {device.is_on()})")
        
        # Test toggle
        device.toggle()
        print(f"✓ Device toggled ON (is_on: {device.is_on()})")
        
        device.toggle()
        print(f"✓ Device toggled OFF (is_on: {device.is_on()})")
        
        # Test all off
        device.on()
        device.all_off()
        print(f"✓ All channels turned OFF (is_on: {device.is_on()})")
        
        # Test reset
        device.on()
        device.reset()
        print(f"✓ Device reset (is_on: {device.is_on()})")
        
        device.disconnect()
        print("✓ Simulated device disconnected")
        
        return True
    except Exception as e:
        print(f"✗ Simulated device test failed: {e}")
        return False


def test_excitation_axis():
    """Test the ExcitationAxis for multi-axis scans."""
    print("\nTesting ExcitationAxis...")
    
    try:
        # Create simulated device
        device = SimulatedExcitationDevice("TestExcitation")
        device.connect()
        
        # Test with default ON/OFF pattern
        axis = ExcitationAxis(device, states=[True, False], wait_s=0.1)
        print(f"✓ ExcitationAxis created with states: {axis.states}")
        
        # Test preparation
        axis.prepare()
        print(f"✓ Axis prepared (device state: {device.is_on()})")
        
        # Test positions
        positions = list(axis.positions())
        print(f"✓ Axis positions: {positions}")
        
        # Test applying states
        for pos in positions:
            axis.apply(pos)
            print(f"✓ Applied state {pos} (device is_on: {device.is_on()})")
        
        # Test state updates
        for pos in positions:
            updates = axis.state_updates(pos)
            print(f"✓ State updates for {pos}: {updates}")
        
        # Test with ON only pattern
        axis_on = ExcitationAxis(device, states=[True], wait_s=0.0)
        axis_on.prepare()
        print(f"✓ ON-only axis prepared (device is_on: {device.is_on()})")
        
        # Test with OFF only pattern
        axis_off = ExcitationAxis(device, states=[False], wait_s=0.0)
        axis_off.prepare()
        print(f"✓ OFF-only axis prepared (device is_on: {device.is_on()})")
        
        device.disconnect()
        print("✓ ExcitationAxis test completed")
        
        return True
    except Exception as e:
        print(f"✗ ExcitationAxis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_excitation_device_interface():
    """Test that ExcitationDevice implements the ExcitationSource interface."""
    print("\nTesting ExcitationDevice interface compliance...")
    
    try:
        # Check that SimulatedExcitationDevice has required methods
        device = SimulatedExcitationDevice("TestExcitation")
        
        required_methods = ['connect', 'disconnect', 'reset', 'get_capabilities', 
                          'on', 'off', 'set_channel', 'get_channel']
        
        all_present = True
        for method in required_methods:
            if hasattr(device, method):
                print(f"✓ Method {method} exists")
            else:
                print(f"✗ Method {method} missing")
                all_present = False
        
        # Check that it's an instance of ExcitationSource
        if isinstance(device, ExcitationSource):
            print("✓ SimulatedExcitationDevice is an ExcitationSource")
        else:
            print("✗ SimulatedExcitationDevice is not an ExcitationSource")
            all_present = False
        
        return all_present
    except Exception as e:
        print(f"✗ Interface test failed: {e}")
        return False


def test_axis_config():
    """Test that AxisConfig can be created for ExcitationAxis."""
    print("\nTesting ExcitationAxis configuration...")
    
    try:
        # Create an AxisConfig like the dialog would
        config = AxisConfig(
            axis_type="Excitation",
            params={
                "states": [True, False, True, False],
                "wait": 0.1,
                "channel": 0,
                "excitation": "TestExcitation"
            }
        )
        
        print(f"✓ AxisConfig created: {config}")
        print(f"✓ Axis type: {config.axis_type}")
        print(f"✓ Parameters: {config.params}")
        
        return True
    except Exception as e:
        print(f"✗ AxisConfig test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Excitation Device and Axis Tests")
    print("=" * 60)
    
    results = []
    results.append(test_simulated_excitation_device())
    results.append(test_excitation_axis())
    results.append(test_excitation_device_interface())
    results.append(test_axis_config())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All tests passed")
        print("\nExcitation Device Functionality:")
        print("- Device control: ON/OFF, channel selection, toggle, all off")
        print("- Multi-axis integration: ExcitationAxis for scan sequences")
        print("- Interface compliance: Implements ExcitationSource base class")
        print("- Configuration: AxisConfig support for GUI integration")
        print("- Simulation: SimulatedExcitationDevice for testing without hardware")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())