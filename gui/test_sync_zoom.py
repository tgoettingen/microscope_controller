"""
Test script to debug zoom sync issue when adding new images.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


def test_zoom_sync_logic():
    """Test the zoom sync logic without GUI."""
    print("=" * 60)
    print("Testing Zoom Sync Logic (No GUI)")
    print("=" * 60)
    
    # Create QApplication first
    from PyQt6.QtWidgets import QApplication
    if QApplication.instance() is None:
        _ = QApplication(sys.argv)
    
    from gui.multi_image_display import MultiImageDisplay
    
    # Create display
    display = MultiImageDisplay()
    
    # Create test images
    image1 = np.random.rand(100, 100)
    image2 = np.random.rand(100, 100)
    
    print("\n1. Adding first image...")
    display.set_image("Image 1", image1)
    print(f"   Images: {list(display._images.keys())}")
    print(f"   Plot widgets: {list(display._plot_widgets.keys())}")
    
    print("\n2. Simulating widget creation (show_display)...")
    # Manually simulate what show_display does
    display._import_pyqt()
    display._create_display_window()
    display.auto_assign_rgb_colormaps()
    display._update_all_images()
    
    # Check initial range
    if display._plot_widgets:
        first_plot = list(display._plot_widgets.values())[0]
        initial_range = first_plot.plotItem.vb.viewRange()
        print(f"   Initial range: X=[{initial_range[0][0]:.2f}, {initial_range[0][1]:.2f}], Y=[{initial_range[1][0]:.2f}, {initial_range[1][1]:.2f}]")
    
    print("\n3. Simulating zoom in...")
    if display._plot_widgets:
        first_plot = list(display._plot_widgets.values())[0]
        # Zoom in to a smaller range
        first_plot.setXRange(25, 75)
        first_plot.setYRange(25, 75)
        zoomed_range = first_plot.plotItem.vb.viewRange()
        print(f"   Zoomed range: X=[{zoomed_range[0][0]:.2f}, {zoomed_range[0][1]:.2f}], Y=[{zoomed_range[1][0]:.2f}, {zoomed_range[1][1]:.2f}]")
    
    print("\n4. Storing existing range before adding new image...")
    existing_range = None
    if display._plot_widgets:
        first_plot = list(display._plot_widgets.values())[0]
        existing_range = first_plot.plotItem.vb.viewRange()
        print(f"   Stored range: X=[{existing_range[0][0]:.2f}, {existing_range[0][1]:.2f}], Y=[{existing_range[1][0]:.2f}, {existing_range[1][1]:.2f}]")
    
    print("\n5. Adding second image...")
    display.set_image("Image 2", image2)
    print(f"   Images: {list(display._images.keys())}")
    
    print("\n6. Creating new plot widgets (simulating recreate)...")
    # This is where the issue might occur
    display._create_plot_widgets()
    
    print(f"   Plot widgets after recreate: {list(display._plot_widgets.keys())}")
    
    # Check if range was preserved
    print("\n7. Checking ranges after widget recreation...")
    if display._plot_widgets:
        for name, plot_widget in display._plot_widgets.items():
            current_range = plot_widget.plotItem.vb.viewRange()
            print(f"   Range for '{name}': X=[{current_range[0][0]:.2f}, {current_range[0][1]:.2f}], Y=[{current_range[1][0]:.2f}, {current_range[1][1]:.2f}]")
            
            # Check if ranges match
            if existing_range:
                if current_range[0] != existing_range[0] or current_range[1] != existing_range[1]:
                    print(f"   WARNING: Range changed for '{name}'!")
                    print(f"   Expected: X=[{existing_range[0][0]:.2f}, {existing_range[0][1]:.2f}], Y=[{existing_range[1][0]:.2f}, {existing_range[1][1]:.2f}]")
                    print(f"   Got:      X=[{current_range[0][0]:.2f}, {current_range[0][1]:.2f}], Y=[{current_range[1][0]:.2f}, {current_range[1][1]:.2f}]")
                else:
                    print(f"   OK: Range preserved for '{name}'")
    
    print("\n8. Testing sync limits flag...")
    print(f"   Sync limits enabled: {display._sync_limits}")
    
    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)
    
    # Cleanup
    display.cleanup()
    
    return display


if __name__ == "__main__":
    test_zoom_sync_logic()
