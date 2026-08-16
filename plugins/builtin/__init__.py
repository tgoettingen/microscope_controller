"""
Built-in plugins for microscope controller.
"""

from .peak_finder import PeakFinderPlugin
from .threshold_analyzer import ThresholdAnalyzerPlugin

__all__ = ['PeakFinderPlugin', 'ThresholdAnalyzerPlugin']
