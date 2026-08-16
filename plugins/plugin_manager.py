"""
Plugin manager for loading and managing microscope controller plugins.
"""

import importlib.util
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
import json
import logging

from .base_plugin import BasePlugin, PluginData, PluginResult

logger = logging.getLogger(__name__)


class PluginManager:
    """Manages plugin loading, configuration, and execution."""
    
    def __init__(self):
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugin_paths: List[Path] = []
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
        
        # Default plugin directories
        self._default_plugin_dirs = [
            Path(__file__).parent / "builtin",  # Built-in plugins
            Path.home() / ".microscope_controller" / "plugins",  # User plugins
        ]
    
    def add_plugin_directory(self, directory: Path) -> None:
        """Add a directory to search for plugins.
        
        Args:
            directory: Path to plugin directory
        """
        if directory.exists() and directory.is_dir():
            self._plugin_paths.append(directory)
            logger.info(f"Added plugin directory: {directory}")
        else:
            logger.warning(f"Plugin directory does not exist: {directory}")
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins in plugin directories.
        
        Returns:
            List of discovered plugin names
        """
        discovered = []
        
        search_dirs = self._default_plugin_dirs + self._plugin_paths
        
        for plugin_dir in search_dirs:
            if not plugin_dir.exists():
                continue
                
            for plugin_file in plugin_dir.glob("*.py"):
                # Skip files that should not be loaded as plugins
                if plugin_file.name.startswith("_"):
                    continue
                if plugin_file.name == "plugin_manager.py":
                    continue
                if plugin_file.name == "base_plugin.py":
                    continue
                    
                try:
                    module_name = plugin_file.stem
                    discovered.append(module_name)
                    logger.info(f"Discovered plugin: {module_name} (from {plugin_dir})")
                except Exception as e:
                    logger.warning(f"Error discovering plugin {plugin_file}: {e}")
        
        return discovered
    
    def auto_load_custom_plugins(self) -> int:
        """Auto-load all plugins from custom directories.
        
        Returns:
            Number of plugins successfully loaded
        """
        loaded_count = 0
        
        # Get custom plugin directories (excluding built-in)
        custom_dirs = self._plugin_paths
        
        for plugin_dir in custom_dirs:
            if not plugin_dir.exists():
                continue
                
            for plugin_file in plugin_dir.glob("*.py"):
                # Skip files that should not be loaded as plugins
                if plugin_file.name.startswith("_"):
                    continue
                if plugin_file.name == "plugin_manager.py":
                    continue
                if plugin_file.name == "base_plugin.py":
                    continue
                    
                try:
                    plugin_name = plugin_file.stem
                    success = self.load_plugin(plugin_name, plugin_file)
                    if success:
                        loaded_count += 1
                        logger.info(f"Auto-loaded plugin: {plugin_name}")
                    else:
                        logger.warning(f"Failed to auto-load plugin: {plugin_name}")
                except Exception as e:
                    logger.warning(f"Error auto-loading plugin {plugin_file}: {e}")
        
        return loaded_count
    
    def load_plugin(self, plugin_name: str, plugin_path: Optional[Path] = None) -> bool:
        """Load a plugin by name or path.
        
        Args:
            plugin_name: Name of the plugin
            plugin_path: Optional path to plugin file
            
        Returns:
            True if loading successful, False otherwise
        """
        try:
            if plugin_path:
                # Load from specific path
                # Create a unique module name to avoid conflicts
                module_name = f"custom_plugin_{plugin_name}_{int(time.time())}"
                spec = importlib.util.spec_from_file_location(module_name, plugin_path)
                if spec is None or spec.loader is None:
                    logger.error(f"Cannot load plugin spec from {plugin_path}")
                    return False
                    
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # Use the original plugin_name for storage
                storage_name = plugin_name
            else:
                # Search for plugin in default directories
                search_dirs = self._default_plugin_dirs + self._plugin_paths
                plugin_file = None
                
                for plugin_dir in search_dirs:
                    potential_file = plugin_dir / f"{plugin_name}.py"
                    if potential_file.exists():
                        plugin_file = potential_file
                        break
                
                if plugin_file is None:
                    logger.error(f"Plugin file not found: {plugin_name}")
                    return False
                    
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
                if spec is None or spec.loader is None:
                    logger.error(f"Cannot load plugin spec from {plugin_file}")
                    return False
                    
                module = importlib.util.module_from_spec(spec)
                sys.modules[plugin_name] = module
                spec.loader.exec_module(module)
                storage_name = plugin_name
            
            # Find plugin class in module
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    attr_name.endswith('Plugin') and
                    attr.__name__.endswith('Plugin') and
                    attr.__name__ != 'BasePlugin' and
                    attr.__name__ != 'DecoderPlugin' and
                    attr.__name__ != 'TimeSeriesPlugin' and
                    attr.__name__ != 'MovementPlugin'):
                    # Check if it has the required methods
                    if (hasattr(attr, 'get_name') and 
                        hasattr(attr, 'get_description') and 
                        hasattr(attr, 'get_version')):
                        plugin_class = attr
                        logger.info(f"Found plugin class: {attr_name}")
                        break
            
            if plugin_class is None:
                # Try the old method as fallback
                try:
                    from plugins.base_plugin import BasePlugin
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, BasePlugin) and 
                            attr != BasePlugin and
                            not attr.__name__ in ['BasePlugin', 'DecoderPlugin', 'TimeSeriesPlugin', 'MovementPlugin']):
                            plugin_class = attr
                            logger.info(f"Found plugin class via BasePlugin: {attr_name}")
                            break
                except Exception as e:
                    logger.debug(f"BasePlugin import failed: {e}")
            
            if plugin_class is None:
                logger.error(f"No valid plugin class found in {plugin_name}")
                logger.debug(f"Module attributes: {[attr for attr in dir(module) if not attr.startswith('_')]}")
                return False
            
            # Instantiate plugin
            plugin_instance = plugin_class()
            
            # Load saved configuration if available
            config = self._plugin_configs.get(storage_name, {})
            if config:
                plugin_instance.initialize(config)
            
            self._plugins[storage_name] = plugin_instance
            logger.info(f"Successfully loaded plugin: {storage_name}")
            return True
            
        except Exception as e:
            logger.exception(f"Error loading plugin {plugin_name}: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin.
        
        Args:
            plugin_name: Name of plugin to unload
            
        Returns:
            True if successful, False otherwise
        """
        if plugin_name not in self._plugins:
            logger.warning(f"Plugin not loaded: {plugin_name}")
            return False
        
        try:
            plugin = self._plugins[plugin_name]
            plugin.cleanup()
            del self._plugins[plugin_name]
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        except Exception as e:
            logger.exception(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a loaded plugin instance.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(plugin_name)
    
    def get_all_plugins(self) -> Dict[str, BasePlugin]:
        """Get all loaded plugins.
        
        Returns:
            Dictionary mapping plugin names to instances
        """
        return self._plugins.copy()
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin.
        
        Args:
            plugin_name: Name of plugin to enable
            
        Returns:
            True if successful, False otherwise
        """
        plugin = self.get_plugin(plugin_name)
        if plugin:
            plugin.enabled = True
            logger.info(f"Enabled plugin: {plugin_name}")
            return True
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin.
        
        Args:
            plugin_name: Name of plugin to disable
            
        Returns:
            True if successful, False otherwise
        """
        plugin = self.get_plugin(plugin_name)
        if plugin:
            plugin.enabled = False
            logger.info(f"Disabled plugin: {plugin_name}")
            return True
        return False
    
    def configure_plugin(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Configure a plugin.
        
        Args:
            plugin_name: Name of plugin
            config: Configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            logger.error(f"Plugin not found: {plugin_name}")
            return False
        
        try:
            # Validate configuration
            is_valid, error_msg = plugin.validate_config(config)
            if not is_valid:
                logger.error(f"Invalid configuration for {plugin_name}: {error_msg}")
                return False
            
            # Apply configuration
            success = plugin.initialize(config)
            if success:
                self._plugin_configs[plugin_name] = config
                logger.info(f"Configured plugin: {plugin_name}")
            return success
        except Exception as e:
            logger.exception(f"Error configuring plugin {plugin_name}: {e}")
            return False
    
    def process_data_with_plugins(self, data: PluginData, plugin_names: Optional[List[str]] = None) -> Dict[str, PluginResult]:
        """Process data with specified plugins.
        
        Args:
            data: Input measurement data
            plugin_names: List of plugin names to use (None = all enabled plugins)
            
        Returns:
            Dictionary mapping plugin names to their results
        """
        results = {}
        
        if plugin_names is None:
            # Use all enabled plugins
            plugin_names = [name for name, plugin in self._plugins.items() if plugin.enabled]
        
        for plugin_name in plugin_names:
            plugin = self.get_plugin(plugin_name)
            if not plugin or not plugin.enabled:
                continue
            
            try:
                result = plugin.process_data(data)
                results[plugin_name] = result
                logger.debug(f"Plugin {plugin_name} processed data successfully")
            except Exception as e:
                logger.exception(f"Error processing data with plugin {plugin_name}: {e}")
                results[plugin_name] = PluginResult(success=False, message=str(e))
        
        return results
    
    def get_movement_commands(self, data: PluginData, plugin_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get movement commands from plugins.
        
        Args:
            data: Input measurement data
            plugin_names: List of plugin names to query (None = all enabled plugins)
            
        Returns:
            List of movement command dictionaries
        """
        all_commands = []
        
        if plugin_names is None:
            plugin_names = [name for name, plugin in self._plugins.items() if plugin.enabled]
        
        for plugin_name in plugin_names:
            plugin = self.get_plugin(plugin_name)
            if not plugin or not plugin.enabled:
                continue
            
            try:
                # First process data to get results
                result = plugin.process_data(data)
                
                # Check if plugin wants to trigger movement
                if plugin.should_trigger_movement(data, result):
                    commands = plugin.get_movement_commands(data, result)
                    all_commands.extend(commands)
                    logger.info(f"Plugin {plugin_name} generated {len(commands)} movement commands")
            except Exception as e:
                logger.exception(f"Error getting movement commands from plugin {plugin_name}: {e}")
        
        return all_commands
    
    def save_plugin_configs(self, file_path: Path) -> bool:
        """Save all plugin configurations to a file.
        
        Args:
            file_path: Path to save configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_data = {
                "plugins": self._plugin_configs,
                "enabled_plugins": [name for name, plugin in self._plugins.items() if plugin.enabled]
            }
            
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved plugin configurations to {file_path}")
            return True
        except Exception as e:
            logger.exception(f"Error saving plugin configurations: {e}")
            return False
    
    def load_plugin_configs(self, file_path: Path) -> bool:
        """Load plugin configurations from a file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
            
            # Load configurations
            if "plugins" in config_data:
                self._plugin_configs.update(config_data["plugins"])
            
            # Set enabled state
            if "enabled_plugins" in config_data:
                for plugin_name in config_data["enabled_plugins"]:
                    self.enable_plugin(plugin_name)
            
            logger.info(f"Loaded plugin configurations from {file_path}")
            return True
        except Exception as e:
            logger.exception(f"Error loading plugin configurations: {e}")
            return False
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a plugin.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            Dictionary with plugin information or None if not found
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            return None
        
        return {
            "name": plugin.get_name(),
            "description": plugin.get_description(),
            "version": plugin.get_version(),
            "author": plugin.author,
            "enabled": plugin.enabled,
            "required_detectors": plugin.get_required_detectors(),
            "required_axes": plugin.get_required_axes(),
            "config_schema": plugin.get_config_schema(),
            "current_config": self._plugin_configs.get(plugin_name, {})
        }
    
    def get_all_plugin_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded plugins.
        
        Returns:
            Dictionary mapping plugin names to their information
        """
        return {name: self.get_plugin_info(name) for name in self._plugins.keys()}


# Global plugin manager instance
_plugin_manager = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager