"""Test script to verify calibration dialog changes."""
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

def test_imports():
    """Test that the module can be imported."""
    try:
        from gui.dialogs.stage_calibration_dialog import StageCalibrationDialog
        print("✓ Successfully imported StageCalibrationDialog")
        return True
    except Exception as e:
        print(f"✗ Failed to import StageCalibrationDialog: {e}")
        return False

def test_method_existence():
    """Test that new methods exist."""
    try:
        from gui.dialogs.stage_calibration_dialog import StageCalibrationDialog
        
        # Check that new methods exist
        methods = [
            '_show_save_options_dialog',
            '_confirm_overwrite', 
            '_get_new_config_path',
            '_perform_save'
        ]
        
        for method in methods:
            if hasattr(StageCalibrationDialog, method):
                print(f"✓ Method {method} exists")
            else:
                print(f"✗ Method {method} missing")
                return False
                
        return True
    except Exception as e:
        print(f"✗ Error checking methods: {e}")
        return False

def main():
    print("Testing calibration dialog changes...")
    print("-" * 50)
    
    results = []
    results.append(test_imports())
    results.append(test_method_existence())
    
    print("-" * 50)
    if all(results):
        print("✓ All tests passed")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())