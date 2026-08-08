"""Advanced test script for calibration dialog functionality."""
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

def test_filename_validation():
    """Test filename validation logic."""
    print("Testing filename validation...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_config_path = Path(tmpdir) / "test_config.json"
        
        # Create a test config file
        test_config = {"stage": {"scaling": {"x_scale": 1.0, "y_scale": 1.0}}}
        with open(mock_config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Test valid filenames
        valid_names = [
            "calibration_new.json",
            "config_v2.json",
            "test_123.json"
        ]
        
        for name in valid_names:
            # Check basic validation logic
            if not name.endswith('.json'):
                print(f"✗ {name} should end with .json")
                continue
                
            invalid_chars = '<>:"/\\|?*'
            if any(char in name for char in invalid_chars):
                print(f"✗ {name} contains invalid characters")
                continue
                
            test_path = mock_config_path.parent / name
            if test_path.exists():
                print(f"✗ {name} already exists")
                continue
                
            print(f"✓ {name} is valid")
        
        # Test invalid filenames
        invalid_names = [
            "",  # empty
            "no_extension",  # no .json
            "test.txt",  # wrong extension
            "test<.json",  # invalid character
            "test:.json",  # invalid character
        ]
        
        for name in invalid_names:
            if not name:
                print(f"✓ Empty filename correctly identified as invalid")
            elif not name.endswith('.json'):
                print(f"✓ '{name}' correctly identified as invalid (no .json extension)")
            else:
                invalid_chars = '<>:"/\\|?*'
                if any(char in name for char in invalid_chars):
                    print(f"✓ '{name}' correctly identified as invalid (contains invalid characters)")
                else:
                    print(f"? '{name}' validation unclear")
    
    print("✓ Filename validation tests completed")

def test_save_methods_callable():
    """Test that save methods are callable."""
    print("Testing save methods callability...")
    
    # Import just the module, not the widget class
    import gui.dialogs.stage_calibration_dialog as cal_module
    
    # Test that the module has the expected methods
    methods = [
        '_show_save_options_dialog',
        '_confirm_overwrite',
        '_get_new_config_path',
        '_perform_save'
    ]
    
    # Check if the StageCalibrationDialog class has these methods
    if hasattr(cal_module, 'StageCalibrationDialog'):
        dialog_class = cal_module.StageCalibrationDialog
        for method_name in methods:
            if hasattr(dialog_class, method_name):
                method = getattr(dialog_class, method_name)
                if callable(method):
                    print(f"✓ Method {method_name} is callable")
                else:
                    print(f"✗ Method {method_name} is not callable")
            else:
                print(f"✗ Method {method_name} not found")
    else:
        print("✗ StageCalibrationDialog class not found")
    
    print("✓ Save methods callability tests completed")

def test_datetime_import():
    """Test that datetime module is properly imported."""
    print("Testing datetime import...")
    
    try:
        import gui.dialogs.stage_calibration_dialog as cal_module
        # Check if datetime is available in the module
        import inspect
        source = inspect.getsource(cal_module)
        if 'from datetime import datetime' in source or 'import datetime' in source:
            print(f"✓ datetime module imported in calibration dialog")
            # Test that we can create a timestamp
            test_time = datetime.now()
            print(f"✓ Can create timestamp: {test_time.isoformat()}")
        else:
            print(f"✗ datetime module not found in source")
    except Exception as e:
        print(f"✗ Failed to check datetime import: {e}")
    
    print("✓ DateTime import test completed")

def test_logging_functionality():
    """Test that logging is properly configured."""
    print("Testing logging functionality...")
    
    try:
        import gui.dialogs.stage_calibration_dialog as cal_module
        if hasattr(cal_module, 'logger'):
            logger = cal_module.logger
            print(f"✓ Logger available: {logger}")
            print(f"✓ Logger name: {logger.name}")
        else:
            print(f"✗ Logger not found in module")
    except Exception as e:
        print(f"✗ Failed to access logger: {e}")
    
    print("✓ Logging functionality test completed")

def main():
    print("Advanced calibration dialog testing...")
    print("=" * 60)
    
    test_filename_validation()
    print()
    test_save_methods_callable()
    print()
    test_datetime_import()
    print()
    test_logging_functionality()
    
    print("=" * 60)
    print("✓ All advanced tests completed")

if __name__ == "__main__":
    main()