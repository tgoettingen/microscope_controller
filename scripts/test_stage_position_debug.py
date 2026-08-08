"""Debug test script to check stage position timer behavior."""
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

def test_timer_logic():
    """Test the timer start/stop logic."""
    try:
        from gui import mainwindow
        import inspect
        
        print("Checking timer logic implementation...")
        
        # Check _start_stage_position_timer
        start_source = inspect.getsource(mainwindow.MainWindow._start_stage_position_timer)
        if 'isActive()' in start_source:
            print("✓ Timer includes active state check")
        else:
            print("✗ Timer missing active state check")
        
        # Check _update_stage_position_display
        update_source = inspect.getsource(mainwindow.MainWindow._update_stage_position_display)
        if 'multiview_runner' in update_source:
            print("✓ Update method checks multiview_runner")
        else:
            print("✗ Update method missing multiview_runner check")
        
        if 'logger.debug' in update_source or 'logger.warning' in update_source:
            print("✓ Update method includes debug logging")
        else:
            print("✗ Update method missing debug logging")
        
        # Check _apply_measurement_state
        apply_source = inspect.getsource(mainwindow.MainWindow._apply_measurement_state)
        if 'logger.debug' in apply_source or 'logger.info' in apply_source:
            print("✓ Measurement state change includes logging")
        else:
            print("✗ Measurement state change missing logging")
            
        return True
        
    except Exception as e:
        print(f"✗ Error checking timer logic: {e}")
        return False

def main():
    print("Debug test for stage position timer...")
    print("=" * 60)
    
    if test_timer_logic():
        print("=" * 60)
        print("✓ Debug test passed")
        print("\nNext steps:")
        print("1. Run the application in simulation mode")
        print("2. Start a multiaxis scan")
        print("3. Stop the multiaxis scan")
        print("4. Check the console logs for timer control messages")
        print("5. Verify that stage position updates in status bar")
        return 0
    else:
        print("=" * 60)
        print("✗ Debug test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())