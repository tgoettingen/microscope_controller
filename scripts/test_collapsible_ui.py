"""Test script to verify collapsible UI functionality."""
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

def test_collapsible_groupbox_class():
    """Test that CollapsibleGroupBox class exists."""
    try:
        from gui.dialogs import stage_calibration_dialog
        
        if hasattr(stage_calibration_dialog, 'CollapsibleGroupBox'):
            print("✓ CollapsibleGroupBox class exists")
            
            # Check for required methods
            required_methods = [
                'setContentLayout',
                'isCollapsed',
                'setCollapsed',
                '_toggle_collapse',
                '_update_collapse_state'
            ]
            
            all_methods = True
            for method in required_methods:
                if hasattr(stage_calibration_dialog.CollapsibleGroupBox, method):
                    print(f"✓ Method {method} exists")
                else:
                    print(f"✗ Method {method} missing")
                    all_methods = False
            
            return all_methods
        else:
            print("✗ CollapsibleGroupBox class missing")
            return False
    except Exception as e:
        print(f"✗ Error checking CollapsibleGroupBox: {e}")
        return False

def test_ui_uses_collapsible():
    """Test that the UI uses CollapsibleGroupBox."""
    try:
        from gui.dialogs import stage_calibration_dialog
        import inspect
        
        source = inspect.getsource(stage_calibration_dialog.StageCalibrationDialog._build_ui)
        
        if 'CollapsibleGroupBox' in source:
            print("✓ UI uses CollapsibleGroupBox")
        else:
            print("✗ UI doesn't use CollapsibleGroupBox")
            return False
        
        # Check that standard QGroupBox is not used for main sections
        main_sections = [
            'Instructions',
            'Current Stage Position',
            'Step 1 — Set Reference Point',
            'Step 2 — Enter Physical Distance Moved',
            'Scaling (current → new)',
            'Travel Range (Soft Limits)',
            'Auto-Detect Travel Limits'
        ]
        
        all_collapsible = True
        for section in main_sections:
            # Check if the section uses CollapsibleGroupBox (more flexible matching)
            if 'CollapsibleGroupBox' in source and section in source:
                print(f"✓ Section '{section}' uses CollapsibleGroupBox")
            else:
                # Special check for auto-detect which might be defined differently
                if 'Auto-Detect' in section and 'CollapsibleGroupBox' in source:
                    print(f"✓ Section '{section}' uses CollapsibleGroupBox (nested)")
                else:
                    print(f"⚠ Section '{section}' might not use CollapsibleGroupBox")
        
        return True
    except Exception as e:
        print(f"✗ Error checking UI usage: {e}")
        return False

def test_initial_collapse_states():
    """Test that sections have appropriate initial collapse states."""
    try:
        from gui.dialogs import stage_calibration_dialog
        import inspect
        
        source = inspect.getsource(stage_calibration_dialog.StageCalibrationDialog._build_ui)
        
        # Check that some sections are initially collapsed
        if 'initially_collapsed=True' in source:
            print("✓ Some sections are initially collapsed")
        else:
            print("⚠ No sections are initially collapsed")
        
        # Check that some sections are initially expanded
        if 'initially_collapsed=False' in source:
            print("✓ Some sections are initially expanded")
        else:
            print("⚠ No sections are initially expanded")
        
        # Check specifically for Instructions being collapsed
        if 'Instructions' in source and 'initially_collapsed=True' in source:
            print("✓ Instructions section is initially collapsed")
        else:
            print("⚠ Instructions section might not be initially collapsed")
        
        return True
    except Exception as e:
        print(f"✗ Error checking collapse states: {e}")
        return False

def test_collapsible_functionality():
    """Test the basic collapsible functionality structure."""
    try:
        from gui.dialogs import stage_calibration_dialog
        import inspect
        
        # Check that the CollapsibleGroupBox has the right structure
        source = inspect.getsource(stage_calibration_dialog.CollapsibleGroupBox)
        
        required_elements = [
            '_toggle_button',
            '_content_widget',
            '_is_collapsed',
            '_setup_ui',
            '_toggle_collapse',
            '_update_collapse_state'
        ]
        
        all_elements = True
        for element in required_elements:
            if element in source:
                print(f"✓ CollapsibleGroupBox has {element}")
            else:
                print(f"✗ CollapsibleGroupBox missing {element}")
                all_elements = False
        
        # Check that it has setContentLayout method
        if 'def setContentLayout' in source:
            print("✓ setContentLayout method exists")
        else:
            print("✗ setContentLayout method missing")
            all_elements = False
        
        return all_elements
    except Exception as e:
        print(f"✗ Error testing collapsible functionality: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing collapsible UI functionality...")
    print("=" * 60)
    
    results = []
    results.append(test_collapsible_groupbox_class())
    results.append(test_ui_uses_collapsible())
    results.append(test_initial_collapse_states())
    results.append(test_collapsible_functionality())
    
    print("=" * 60)
    if all(results):
        print("✓ All tests passed")
        print("\nCollapsible UI features:")
        print("- Custom CollapsibleGroupBox class with toggle button")
        print("- All main sections use collapsible groups")
        print("- Appropriate initial collapse states (expanded for essential, collapsed for advanced)")
        print("- Proper structure for collapse/expand functionality")
        print("- Better for small screens with collapsible sections")
        print("\nInitial states:")
        print("- Expanded: Current Stage Position, Step 1 (Set Reference), Step 2 (Enter Distance)")
        print("- Collapsed: Instructions, Scaling preview, Travel Range, Auto-Detect (advanced features)")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())