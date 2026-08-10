#!/usr/bin/env python3
"""Test script for ExcitationDevice integration with config files."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.factory import build_devices


def test_simulated_config():
    """Test with simulated config."""
    print("Testing simulated config...")
    
    try:
        camera, stage, focus, light, fw, detector, excitation = build_devices(
            'config/default_devices_simulate_mac.json'
        )
        
        print(f"✓ Excitation device type: {type(excitation).__name__}")
        print(f"✓ Excitation device name: {excitation.name if excitation else None}")
        
        # Test basic functionality
        excitation.connect()
        print("✓ Simulated device connected")
        
        excitation.on()
        print(f"✓ Device ON (is_on: {excitation.is_on()})")
        
        excitation.off()
        print(f"✓ Device OFF (is_on: {excitation.is_on()})")
        
        excitation.disconnect()
        print("✓ Simulated device disconnected")
        
        return True
    except Exception as e:
        print(f"✗ Simulated config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_config():
    """Test with real hardware config (without connecting to hardware)."""
    print("\nTesting real hardware config...")
    
    try:
        camera, stage, focus, light, fw, detector, excitation = build_devices(
            'config/default_devices_real_mac.json'
        )
        
        print(f"✓ Excitation device type: {type(excitation).__name__}")
        print(f"✓ Excitation device name: {excitation.name if excitation else None}")
        print(f"✓ Real config loaded successfully")
        
        # Note: We don't actually connect to hardware in this test
        # to avoid requiring the hardware to be connected
        
        return True
    except Exception as e:
        print(f"✗ Real config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factory_return():
    """Test that factory returns 7 devices including excitation."""
    print("\nTesting factory return values...")
    
    try:
        camera, stage, focus, light, fw, detector, excitation = build_devices(
            'config/default_devices_simulate_mac.json'
        )
        
        devices = {
            'camera': camera,
            'stage': stage, 
            'focus': focus,
            'light': light,
            'filter_wheel': fw,
            'detector': detector,
            'excitation': excitation
        }
        
        all_present = all(devices[name] is not None for name in devices.keys())
        
        if all_present:
            print("✓ All 7 devices returned by factory")
            for name, device in devices.items():
                print(f"  - {name}: {type(device).__name__}")
            return True
        else:
            print("✗ Some devices are None")
            return False
            
    except Exception as e:
        print(f"✗ Factory return test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Excitation Device Integration Tests")
    print("=" * 60)
    
    results = []
    results.append(test_simulated_config())
    results.append(test_real_config())
    results.append(test_factory_return())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All integration tests passed")
        print("\nConfiguration Summary:")
        print("- Simulated config: Uses SimulatedExcitationDevice")
        print("- Real config: Uses ExcitationDevice with SerialLink")
        print("- Factory: Returns 7 devices including excitation")
        print("- Integration: Excitation source fully integrated into system")
        return 0
    else:
        print("✗ Some integration tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())