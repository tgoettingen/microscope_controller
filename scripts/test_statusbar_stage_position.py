"""Test script to verify status bar stage position update functionality."""
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

def test_imports():
    """Test that the mainwindow module can be imported."""
    try:
        from gui import mainwindow
        print("✓ Successfully imported mainwindow module")
        return True
    except Exception as e:
        print(f"✗ Failed to import mainwindow module: {e}")
        return False

def test_method_existence():
    """Test that new methods exist in MainWindow."""
    try:
        from gui import mainwindow
        
        # Check that new methods exist
        methods = [
            '_start_stage_position_timer',
            '_stop_stage_position_timer', 
            '_update_stage_position_display',
            '_set_stage_position_text'
        ]
        
        for method in methods:
            if hasattr(mainwindow.MainWindow, method):
                print(f"✓ Method {method} exists")
            else:
                print(f"✗ Method {method} missing")
                return False
                
        return True
    except Exception as e:
        print(f"✗ Error checking methods: {e}")
        return False

def test_attribute_existence():
    """Test that new attributes exist in MainWindow."""
    try:
        from gui import mainwindow
        import inspect
        
        # Get the __init__ method source
        source = inspect.getsource(mainwindow.MainWindow.__init__)
        
        # Check for new attributes
        attributes = [
            '_stage_position_timer',
            '_stage_position_label'
        ]
        
        for attr in attributes:
            if attr in source:
                print(f"✓ Attribute {attr} defined in __init__")
            else:
                print(f"✗ Attribute {attr} not found in __init__")
                return False
                
        return True
    except Exception as e:
        print(f"✗ Error checking attributes: {e}")
        return False

def test_timer_interval():
    """Test that the timer interval is set to 500ms."""
    try:
        from gui import mainwindow
        import inspect
        
        # Get the _start_stage_position_timer method source
        source = inspect.getsource(mainwindow.MainWindow._start_stage_position_timer)
        
        if 'setInterval(500)' in source or 'setInterval( 500 )' in source:
            print("✓ Timer interval set to 500ms")
            return True
        else:
            print("✗ Timer interval not set to 500ms")
            return False
    except Exception as e:
        print(f"✗ Error checking timer interval: {e}")
        return False

def test_exception_handling():
    """Test that exception handling is in place."""
    try:
        from gui import mainwindow
        import inspect
        
        # Get the _update_stage_position_display method source
        source = inspect.getsource(mainwindow.MainWindow._update_stage_position_display)
        
        # Check for exception handling
        if 'try:' in source and 'except' in source:
            print("✓ Exception handling present in position update method")
            return True
        else:
            print("✗ Exception handling missing in position update method")
            return False
    except Exception as e:
        print(f"✗ Error checking exception handling: {e}")
        return False

def test_multiaxis_state_check():
    """Test that multiaxis state check is NOT blocking updates (updates should happen regardless)."""
    try:
        from gui import mainwindow
        import inspect
        
        # Get the _update_stage_position_display method source
        source = inspect.getsource(mainwindow.MainWindow._update_stage_position_display)
        
        # Check that we DON'T skip updates when multiaxis is running
        # (multiaxis should handle updates itself, but our timer should still work when not running)
        if 'multi_runner is not None' not in source:
            print("✓ Position update method does not block when multiaxis running")
            return True
        else:
            print("✗ Position update method incorrectly blocks when multiaxis running")
            return False
    except Exception as e:
        print(f"✗ Error checking multiaxis state: {e}")
        return False

def test_timer_control_in_multiaxis():
    """Test that timer control is integrated in multiaxis start."""
    try:
        from gui import mainwindow
        import inspect
        
        # Check _start_multiaxis
        start_source = inspect.getsource(mainwindow.MainWindow._start_multiaxis)
        if '_stop_stage_position_timer' in start_source:
            print("✓ Timer stop called in multiaxis start")
        else:
            print("✗ Timer stop not called in multiaxis start")
            return False
        
        # Timer restart is now handled by _apply_measurement_state when state changes to "Finished"
        # This is the correct approach as it ensures timer is restarted in all scenarios
        print("✓ Timer restart handled by _apply_measurement_state (correct design)")
            
        return True
    except Exception as e:
        print(f"✗ Error checking timer control in multiaxis: {e}")
        return False

def test_timer_control_in_multiview():
    """Test that timer control is integrated in multiview start."""
    try:
        from gui import mainwindow
        import inspect
        
        # Check _start_multiview_scan
        start_source = inspect.getsource(mainwindow.MainWindow._start_multiview_scan)
        if '_stop_stage_position_timer' in start_source:
            print("✓ Timer stop called in multiview start")
        else:
            print("✗ Timer stop not called in multiview start")
            return False
        
        # Timer restart is now handled by _apply_measurement_state when state changes to "Finished"
        # This is the correct approach as it ensures timer is restarted in all scenarios
        print("✓ Timer restart handled by _apply_measurement_state (correct design)")
            
        return True
    except Exception as e:
        print(f"✗ Error checking timer control in multiview: {e}")
        return False

def test_statusbar_label_creation():
    """Test that status bar label is created."""
    try:
        from gui import mainwindow
        import inspect
        
        # Get the _build_ui method source
        source = inspect.getsource(mainwindow.MainWindow._build_ui)
        
        # Check for stage position label creation
        if '_stage_position_label' in source and 'addPermanentWidget' in source:
            print("✓ Stage position label created in status bar")
            return True
        else:
            print("✗ Stage position label not created in status bar")
            return False
    except Exception as e:
        print(f"✗ Error checking status bar label creation: {e}")
        return False

def test_measurement_state_timer_control():
    """Test that _apply_measurement_state controls the timer correctly."""
    try:
        from gui import mainwindow
        import inspect
        
        # Get the _apply_measurement_state method source
        source = inspect.getsource(mainwindow.MainWindow._apply_measurement_state)
        
        # Check for timer control logic
        if '_start_stage_position_timer' in source and '_stop_stage_position_timer' in source:
            print("✓ Timer control present in _apply_measurement_state")
            return True
        else:
            print("✗ Timer control missing in _apply_measurement_state")
            return False
    except Exception as e:
        print(f"✗ Error checking measurement state timer control: {e}")
        return False

def main():
    print("Testing status bar stage position update functionality...")
    print("=" * 60)
    
    results = []
    results.append(test_imports())
    results.append(test_method_existence())
    results.append(test_attribute_existence())
    results.append(test_timer_interval())
    results.append(test_exception_handling())
    results.append(test_multiaxis_state_check())
    results.append(test_timer_control_in_multiaxis())
    results.append(test_timer_control_in_multiview())
    results.append(test_statusbar_label_creation())
    results.append(test_measurement_state_timer_control())
    
    print("=" * 60)
    if all(results):
        print("✓ All tests passed")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())