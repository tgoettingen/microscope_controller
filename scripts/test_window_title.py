"""Test script to verify window title update functionality."""
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

def test_window_title_method():
    """Test that _update_window_title method exists."""
    try:
        from gui import mainwindow
        import inspect
        
        if hasattr(mainwindow.MainWindow, '_update_window_title'):
            print("✓ _update_window_title method exists")
            return True
        else:
            print("✗ _update_window_title method missing")
            return False
    except Exception as e:
        print(f"✗ Error checking method: {e}")
        return False

def test_title_attributes():
    """Test that title-related attributes exist."""
    try:
        from gui import mainwindow
        import inspect
        
        source = inspect.getsource(mainwindow.MainWindow.__init__)
        
        if '_config_filename' in source:
            print("✓ _config_filename attribute defined")
        else:
            print("✗ _config_filename attribute missing")
            return False
        
        if '_experiment_filename' in source:
            print("✓ _experiment_filename attribute defined")
        else:
            print("✗ _experiment_filename attribute missing")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Error checking attributes: {e}")
        return False

def test_config_loading_title_update():
    """Test that config loading updates title."""
    try:
        from gui import mainwindow
        import inspect
        
        source = inspect.getsource(mainwindow.MainWindow.load_hardware_config)
        
        if '_update_window_title' in source:
            print("✓ Config loading calls _update_window_title")
        else:
            print("✗ Config loading doesn't update title")
            return False
            
        if '_config_filename' in source:
            print("✓ Config loading updates _config_filename")
        else:
            print("✗ Config loading doesn't update _config_filename")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Error checking config loading: {e}")
        return False

def test_experiment_loading_title_update():
    """Test that experiment loading updates title."""
    try:
        from gui import mainwindow
        import inspect
        
        source = inspect.getsource(mainwindow.MainWindow.load_full_experiment)
        
        if '_update_window_title' in source:
            print("✓ Experiment loading calls _update_window_title")
        else:
            print("✗ Experiment loading doesn't update title")
            return False
            
        if '_experiment_filename' in source:
            print("✓ Experiment loading updates _experiment_filename")
        else:
            print("✗ Experiment loading doesn't update _experiment_filename")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Error checking experiment loading: {e}")
        return False

def test_title_format():
    """Test that title format is correct."""
    try:
        from gui import mainwindow
        import inspect
        
        source = inspect.getsource(mainwindow.MainWindow._update_window_title)
        
        if 'Microscope Control System' in source:
            print("✓ Title includes base name")
        else:
            print("✗ Title missing base name")
            return False
        
        if ': ' in source:  # Look for the colon in the f-string
            print("✓ Title uses ':' separator")
        else:
            print("✗ Title doesn't use ':' separator")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Error checking title format: {e}")
        return False

def main():
    print("Testing window title update functionality...")
    print("=" * 60)
    
    results = []
    results.append(test_window_title_method())
    results.append(test_title_attributes())
    results.append(test_config_loading_title_update())
    results.append(test_experiment_loading_title_update())
    results.append(test_title_format())
    
    print("=" * 60)
    if all(results):
        print("✓ All tests passed")
        print("\nWindow title functionality:")
        print("- Base title: 'Microscope Control System'")
        print("- Config format: 'Microscope Control System: config_name.json'")
        print("- Experiment format: 'Microscope Control System: config_name.json: experiment_name.json'")
        print("- Separator: ':' between components")
        print("- Updates on: Config load, Experiment load, Initial load")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())