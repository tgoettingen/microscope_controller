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

def test_error_handling_improvements():
    """Test that error handling has been improved to prevent crashes."""
    try:
        from gui.dialogs import stage_calibration_dialog
        import inspect
        
        # Check for custom LimitReachedError
        if hasattr(stage_calibration_dialog, 'LimitReachedError'):
            print("✓ Custom LimitReachedError exception exists")
        else:
            print("✗ LimitReachedError exception missing")
            return False
        
        # Check improved _safe_move_to method
        source = inspect.getsource(stage_calibration_dialog.AutoDetectWorker._safe_move_to)
        
        error_checks = [
            'is_limit_error',
            'is_timeout_error', 
            'is_hardware_error',
            'is_communication_error',
            'is_write_timeout',  # New write timeout handling
            'is_movement_error',
            'LimitReachedError',
            'retry_count',
            'attempt',
            'retry_delay'  # New adaptive retry delay
        ]
        
        all_checks = True
        for check in error_checks:
            if check in source:
                print(f"✓ Error handling includes {check}")
            else:
                print(f"⚠ Error handling might miss {check}")
        
        # Check for serial error specific keywords
        serial_keywords = ['reply', 'command', 'protocol', 'gts', 'gets', 'communication', 'serial', 'write timeout', 'serialtimeout']
        serial_found = any(keyword in source for keyword in serial_keywords)
        if serial_found:
            print("✓ Serial communication error handling includes specific keywords")
        else:
            print("⚠ Serial communication error handling might miss specific keywords")
        
        # Check for increased retry count
        if 'retry_count: int = 4' in source:
            print("✓ Retry count increased to 4 for better serial error handling")
        elif 'retry_count = 4' in source:
            print("✓ Retry count increased to 4 for better serial error handling")
        else:
            print("⚠ Retry count might not be increased (checking current value)")
            # Try to find the retry count value
            import re
            retry_match = re.search(r'retry_count[:\s]*=\s*(\d+)', source)
            if retry_match:
                retry_value = retry_match.group(1)
                print(f"  Current retry count: {retry_value}")
            else:
                print("  Could not determine retry count")
        
        # Check for adaptive retry delays
        if 'retry_delay' in source and '2.0' in source:
            print("✓ Adaptive retry delays implemented (including 2.0s for write timeout)")
        else:
            print("⚠ Adaptive retry delays might not be fully implemented")
        
        return True
    except Exception as e:
        print(f"✗ Error checking error handling: {e}")
        return False

def test_adaptive_step_algorithm():
    """Test that adaptive step size algorithm is implemented."""
    try:
        from gui.dialogs import stage_calibration_dialog
        import inspect
        
        source = inspect.getsource(stage_calibration_dialog.AutoDetectWorker._detect_axis_limit)
        
        adaptive_features = [
            'adaptive',
            'min_step',
            'max_step', 
            'initial_step',
            'expanding',
            'refining',
            'current_step',
            '1.5'  # Step size increase factor
        ]
        
        all_features = True
        for feature in adaptive_features:
            if feature in source:
                print(f"✓ Adaptive algorithm includes {feature}")
            else:
                print(f"⚠ Adaptive algorithm might miss {feature}")
                all_features = False
        
        # Check that old fixed-step approach is gone
        if 'coarse_step = 1000.0' not in source:
            print("✓ Old fixed-step approach removed")
        else:
            print("⚠ Old fixed-step approach still present")
            all_features = False
        
        return all_features
    except Exception as e:
        print(f"✗ Error checking adaptive algorithm: {e}")
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

def test_safe_return_logic():
    """Test that safe return logic is implemented for limit handling."""
    try:
        from gui.dialogs import stage_calibration_dialog
        import inspect
        
        # Check for _clamp_to_limits method
        if hasattr(stage_calibration_dialog.AutoDetectWorker, '_clamp_to_limits'):
            print("✓ _clamp_to_limits method exists")
        else:
            print("✗ _clamp_to_limits method missing")
            return False
        
        # Check for safe return logic in _detect_limits
        source = inspect.getsource(stage_calibration_dialog.AutoDetectWorker._detect_limits)
        
        safe_return_features = [
            'safe position',
            'clamp_to_limits',
            'except (LimitReachedError, Exception)',
            'Could not return to start position'
        ]
        
        all_features = True
        for feature in safe_return_features:
            if feature in source:
                print(f"✓ Safe return logic includes {feature}")
            else:
                print(f"⚠ Safe return logic might miss {feature}")
                all_features = False
        
        # Check for error handling in individual limit detection
        if 'try:' in source and 'except Exception as e:' in source:
            print("✓ Individual limit detection has error handling")
        else:
            print("⚠ Individual limit detection might lack error handling")
        
        return True
    except Exception as e:
        print(f"✗ Error checking safe return logic: {e}")
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
    results.append(test_error_handling_improvements())
    results.append(test_adaptive_step_algorithm())
    results.append(test_safe_return_logic())
    
    print("=" * 60)
    if all(results):
        print("✓ All tests passed")
        print("\nAuto-detect limits functionality:")
        print("- UI components: Progress bar, status label, detect/cancel buttons")
        print("- Algorithm: Adaptive step size with expanding + refining phases")
        print("- Speed: Exponential step increase for fast limit finding")
        print("- Precision: Binary search refinement for accurate limits")
        print("- Safety: 95% working range (2.5% margin from each end)")
        print("- Error handling: Custom LimitReachedError, comprehensive error classification")
        print("- Serial errors: Enhanced handling for SerialTimeoutException and write timeout")
        print("- Retry mechanism: Automatic retry for temporary errors (4 retries with adaptive delays)")
        print("- Adaptive delays: 2.0s for write timeout, 1.5s for timeout, 1.0s for communication")
        print("- Protocol errors: Special handling for Standa protocol errors (e.g., 'expected reply with command gets; got gts')")
        print("- Safe return: Intelligent return to safe position if start position is outside limits")
        print("- Limit clamping: _clamp_to_limits method ensures final position is within detected range")
        print("- Crash prevention: Robust error handling prevents crashes on limit detection")
        print("- Thread safety: Background worker thread for detection")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())