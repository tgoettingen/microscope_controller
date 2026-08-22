"""
Example usage of MultiImageDisplay class
"""

import numpy as np
from multi_image_display import MultiImageDisplay


def example_usage():
    """Demonstrate basic usage of MultiImageDisplay."""
    
    # Create display instance
    display = MultiImageDisplay()
    
    # Create some sample images
    # Image 1: Gaussian blob
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    image1 = np.exp(-(X**2 + Y**2) / 10)
    
    # Image 2: Sine pattern
    image2 = np.sin(X) * np.cos(Y)
    
    # Image 3: Random noise
    image3 = np.random.rand(100, 100)
    
    # Set images (first 3 will get RGB colormaps automatically)
    display.set_image("Gaussian Blob", image1)
    display.set_image("Sine Pattern", image2)
    display.set_image("Random Noise", image3)
    
    # Set channel data for overlay (simulate detector channels)
    display.set_channel_data("Channel 1", image1 * 0.8)
    display.set_channel_data("Channel 2", image2 * 0.6)
    display.set_channel_data("Channel 3", image3 * 0.4)
    
    # Set coordinate ranges
    display.set_coordinate_ranges(-5, 5, -5, 5)
    
    # Show display
    display.show_display()
    
    return display


if __name__ == "__main__":
    display = example_usage()
    print("MultiImageDisplay example running. Close the window to exit.")
