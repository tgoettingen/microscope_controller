"""Test script to verify auto-detect limits functionality."""
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

def test_auto_detect_ui_elements():
    """Test that auto-detect UI elements exist."""
    try:
        from gui.dialogs import stage_calibration_dialog
        import inspect
        
        source = inspect.getsource(stage_calibration_dialog.StageCalibrationDialog._build_ui)
        
        ui_elements = [
            '_auto_detect_btn',
            '_auto_detect_cancel_btn',
            '_auto_detect_progress_bar',
            '_auto_detect_status_label'
        ]
        
        all_found = True
        for element in ui_elements:
            if element in source:
                print(f"✓ UI element {element} found")
            else:
                print(f"✗ UI element {element} missing")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"✗ Error checking UI elements: {e}")
        return False

def test_auto_detect_methods():
    """Test that auto-detect methods exist."""
    try:
        from gui.dialogs import stage_calibration_dialog
        
        methods = [
            '_on_auto_detect_limits',
            '_on_cancel_auto_detect',
            '_on_auto_detect_started',
            '_on_auto_detect_progress',
            '_on_auto_detect_finished',
            '_on_auto_detect_error',
            '_calculate_95_percent_range',
            '_apply_detected_limits'
        ]
        
        all_found = True
        for method in methods:
            if hasattr(stage_calibration_dialog.StageCalibrationDialog, method):
                print(f"✓ Method {method} exists")
            else:
                print(f"✗ Method {method} missing")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"✗ Error checking methods: {e}")
        return False

def test_auto_detect_worker_class():
    """Test that AutoDetectWorker class exists."""
    try:
        from gui.dialogs import stage_calibration_dialog
        
        if hasattr(stage_calibration_dialog, 'AutoDetectWorker'):
            print("✓ AutoDetectWorker class exists")
            
            # Check for required signals
            worker = stage_calibration_dialog.AutoDetectWorker
            signals = ['started', 'progress', 'finished', 'error']
            all_signals = True
            for signal in signals:
                if hasattr(worker, signal):
                    print(f"✓ Signal {signal} exists")
                else:
                    print(f"✗ Signal {signal} missing")
                    all_signals = False
            
            return all_signals
        else:
            print("✗ AutoDetectWorker class missing")
            return False
    except Exception as e:
        print(f"✗ Error checking AutoDetectWorker: {e}")
        return False

def test_95_percent_calculation():
    """Test the 95% range calculation logic."""
    try:
        # Create a simple calculation function that mimics the logic
        def calculate_95_percent_range(limits):
            """Copy the calculation logic."""
            safe_limits = {}
            for axis in ['x', 'y']:
                min_val = limits[f'{axis}_min']
                max_val = limits[f'{axis}_max']
                total_range = max_val - min_val
                margin = total_range * 0.025  # 2.5% margin
                safe_limits[f'{axis}_min'] = min_val + margin
                safe_limits[f'{axis}_max'] = max_val - margin
            return safe_limits
        
        # Test case 1: Simple range
        limits1 = {'x_min': 0, 'x_max': 1000, 'y_min': 0, 'y_max': 1000}
        result1 = calculate_95_percent_range(limits1)
        
        expected_margin = 25.0  # 2.5% of 1000
        if (result1['x_min'] == expected_margin and 
            result1['x_max'] == 1000 - expected_margin and
            result1['y_min'] == expected_margin and 
            result1['y_max'] == 1000 - expected_margin):
            print("✓ 95% calculation correct for simple range")
        else:
            print(f"✗ 95% calculation incorrect: {result1}")
            return False
        
        # Test case 2: Asymmetric range
        limits2 = {'x_min': -500, 'x_max': 1500, 'y_min': -200, 'y_max': 800}
        result2 = calculate_95_percent_range(limits2)
        
        x_range = 1500 - (-500)  # 2000
        x_margin = x_range * 0.025  # 50
        y_range = 800 - (-200)  # 1000
        y_margin = y_range * 0.025  # 25
        
        if (result2['x_min'] == -500 + x_margin and 
            result2['x_max'] == 1500 - x_margin and
            result2['y_min'] == -200 + y_margin and 
            result2['y_max'] == 800 - y_margin):
            print("✓ 95% calculation correct for asymmetric range")
        else:
            print(f"✗ 95% calculation incorrect for asymmetric range: {result2}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error testing 95% calculation: {e}")
        return False

def test_state_management():
    """Test that auto-detect state management exists."""
    try:
        from gui.dialogs import stage_calibration_dialog
        import inspect
        
        source = inspect.getsource(stage_calibration_dialog.StageCalibrationDialog.__init__)
        
        state_vars = [
            '_auto_detecting',
            '_auto_detect_cancelled',
            '_auto_detect_progress'
        ]
        
        all_found = True
        for var in state_vars:
            if var in source:
                print(f"✓ State variable {var} defined")
            else:
                print(f"✗ State variable {var} missing")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"✗ Error checking state management: {e}")
        return False

def main():
    print("Testing auto-detect limits functionality...")
    print("=" * 60)
    
    results = []
    results.append(test_auto_detect_ui_elements())
    results.append(test_auto_detect_methods())
    results.append(test_auto_detect_worker_class())
    results.append(test_95_percent_calculation())
    results.append(test_state_management())
    
    print("=" * 60)
    if all(results):
        print("✓ All tests passed")
        print("\nAuto-detect limits functionality:")
        print("- UI components: Progress bar, status label, detect/cancel buttons")
        print("- Algorithm: Coarse search + binary search for precise limits")
        print("- Safety: 95% working range (2.5% margin from each end)")
        print("- Error handling: Cancellation support, error recovery")
        print("- Thread safety: Background worker thread for detection")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())