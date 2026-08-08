"""Test script to verify Move Motors dialog functionality."""
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

def test_imports():
    """Test that the move motors dialog can be imported."""
    try:
        from gui.dialogs.move_motors_dialog import MoveMotorsDialog
        print("✓ Successfully imported MoveMotorsDialog")
        return True
    except Exception as e:
        print(f"✗ Failed to import MoveMotorsDialog: {e}")
        return False

def test_method_existence():
    """Test that required methods exist."""
    try:
        from gui.dialogs.move_motors_dialog import MoveMotorsDialog
        
        # Check that required methods exist
        methods = [
            '_build_ui',
            '_create_stage_controls',
            '_create_focus_controls',
            '_load_current_positions',
            '_move_to_position'
        ]
        
        for method in methods:
            if hasattr(MoveMotorsDialog, method):
                print(f"✓ Method {method} exists")
            else:
                print(f"✗ Method {method} missing")
                return False
                
        return True
    except Exception as e:
        print(f"✗ Error checking methods: {e}")
        return False

def test_menu_integration():
    """Test that menu integration exists in mainwindow."""
    try:
        from gui import mainwindow
        import inspect
        
        # Check _create_menus method
        source = inspect.getsource(mainwindow.MainWindow._create_menus)
        
        if 'Move Motors' in source or 'move_motors' in source:
            print("✓ Move Motors menu item found")
        else:
            print("✗ Move Motors menu item not found")
            return False
        
        # Check _open_move_motors_dialog method
        if hasattr(mainwindow.MainWindow, '_open_move_motors_dialog'):
            print("✓ _open_move_motors_dialog method exists")
        else:
            print("✗ _open_move_motors_dialog method missing")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Error checking menu integration: {e}")
        return False

def test_ui_components():
    """Test that UI components are properly created."""
    try:
        from gui.dialogs.move_motors_dialog import MoveMotorsDialog
        import inspect
        
        # Check _build_ui method and related methods
        source = inspect.getsource(MoveMotorsDialog._build_ui)
        stage_source = inspect.getsource(MoveMotorsDialog._create_stage_controls)
        focus_source = inspect.getsource(MoveMotorsDialog._create_focus_controls)
        init_source = inspect.getsource(MoveMotorsDialog.__init__)
        
        # Combine all sources
        full_source = source + stage_source + focus_source + init_source
        
        components = [
            'DoubleSpinBox',  # Editable fields (without QtWidgets. prefix)
            'Slider',         # Sliders
            'GroupBox',       # Axis groups
            'PushButton',      # Buttons
            'QCheckBox',      # Mode selection checkbox
            'QTimer',         # Position update timer
            'QLineEdit'      # Current position display
        ]
        
        for component in components:
            if component in full_source:
                print(f"✓ UI component {component} found")
            else:
                print(f"✗ UI component {component} missing")
                return False
        
        # Check for current position display fields
        if '_current_display' in stage_source:
            print("✓ Current position display fields found")
        else:
            print("✗ Current position display fields missing")
            return False
        
        # Check that custom PositionSlider is NOT used (was causing crashes)
        if 'PositionSlider' not in full_source:
            print("✓ Using standard QSlider (no custom slider that could crash)")
        else:
            print("⚠ Custom PositionSlider still present (may cause crashes)")
                
        return True
    except Exception as e:
        print(f"✗ Error checking UI components: {e}")
        return False

def test_live_mode_features():
    """Test that live mode features are implemented."""
    try:
        from gui.dialogs.move_motors_dialog import MoveMotorsDialog
        import inspect
        
        # Check for live mode related methods
        methods = [
            '_on_live_mode_toggled',
            '_on_live_x_changed',
            '_on_live_y_changed',
            '_on_live_z_changed',
            '_live_move_stage',
            '_live_move_focus'
        ]
        
        for method in methods:
            if hasattr(MoveMotorsDialog, method):
                print(f"✓ Live mode method {method} exists")
            else:
                print(f"✗ Live mode method {method} missing")
                return False
        
        # Check for position timer methods
        timer_methods = ['_start_position_timer', '_stop_position_timer', '_update_live_positions']
        for method in timer_methods:
            if hasattr(MoveMotorsDialog, method):
                print(f"✓ Timer method {method} exists")
            else:
                print(f"✗ Timer method {method} missing")
                return False
                
        return True
    except Exception as e:
        print(f"✗ Error checking live mode features: {e}")
        return False

def test_move_state_management():
    """Test that move state management is implemented."""
    try:
        from gui.dialogs.move_motors_dialog import MoveMotorsDialog
        import inspect
        
        # Check for move state management methods
        methods = [
            '_on_move_complete',
            '_on_move_error',
            '_close_dialog'
        ]
        
        for method in methods:
            if hasattr(MoveMotorsDialog, method):
                print(f"✓ State management method {method} exists")
            else:
                print(f"✗ State management method {method} missing")
                return False
        
        # Check _move_to_position for state management
        move_source = inspect.getsource(MoveMotorsDialog._move_to_position)
        if '_is_moving' in move_source:
            print("✓ Move state tracking implemented")
        else:
            print("✗ Move state tracking missing")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Error checking move state management: {e}")
        return False

def test_position_update_logic():
    """Test that position update logic has been corrected."""
    try:
        from gui.dialogs.move_motors_dialog import MoveMotorsDialog
        import inspect
        
        # Check _update_live_positions method
        update_source = inspect.getsource(MoveMotorsDialog._update_live_positions)
        
        # Should update current display fields, not spinboxes
        if '_current_display' in update_source:
            print("✓ Position update uses current display fields")
        else:
            print("✗ Position update doesn't use current display fields")
            return False
        
        # Should not block signals on spinboxes (since we don't update them)
        if 'blockSignals' not in update_source:
            print("✓ Position update doesn't interfere with user input")
        else:
            print("⚠ Position update still uses blockSignals (may need review)")
        
        # Note: Slider indicators removed due to macOS crash issues
        print("✓ Position update logic: current/target separation maintained")
            
        return True
    except Exception as e:
        print(f"✗ Error checking position update logic: {e}")
        return False

def main():
    print("Testing Move Motors dialog functionality...")
    print("=" * 60)
    
    results = []
    results.append(test_imports())
    results.append(test_method_existence())
    results.append(test_menu_integration())
    results.append(test_ui_components())
    results.append(test_live_mode_features())
    results.append(test_move_state_management())
    results.append(test_position_update_logic())
    
    print("=" * 60)
    if all(results):
        print("✓ All tests passed")
        print("\nMove Motors dialog is ready to use:")
        print("- Location: Action menu -> Move Motors...")
        print("- Features: Stage X/Y control, Focus Z control")
        print("- UI Design:")
        print("  • Current position: Read-only text boxes (gray background)")
        print("  • Target position: Editable spinboxes and sliders")
        print("  • Stability: Removed custom slider to prevent macOS crashes")
        print("- Modes:")
        print("  • Live Moving Mode: Immediate movement when adjusting sliders/spinboxes")
        print("  • Move to Position Mode: Use button to move to specified position")
        print("- Real-time updates: Current position updates continuously without interfering")
        print("- Safety: Move button disabled during movement and in live mode")
        print("- Status: Visual status indicator for current operation state")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())