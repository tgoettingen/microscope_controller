"""
Plugin system for microscope controller.

This module provides the base plugin interface and plugin management functionality.
Plugins can process data (images, time series) and control stage movement based on analysis results.
"""

from .base_plugin import BasePlugin, PluginData, PluginResult
from .plugin_manager import PluginManager

__all__ = ['BasePlugin', 'PluginData', 'PluginResult', 'PluginManager']
