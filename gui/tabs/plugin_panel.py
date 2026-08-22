"""
Plugin Panel - GUI for managing microscope controller plugins.

This panel provides:
- List of available plugins
- Enable/disable plugins
- Plugin configuration
- Run plugins on current data
- View plugin status and results
"""

from __future__ import annotations

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QToolTip
from typing import Dict, Any, List, Optional
import json
from pathlib import Path


class PluginPanel(QtWidgets.QWidget):
    """Panel for managing microscope controller plugins."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.parent_window = parent
        self.plugin_manager = None
        self.custom_plugin_dir = None
        self.plugin_configs = {}  # Store plugin configurations
        
        self._setup_ui()
        self._load_plugin_manager()
    
    def _setup_ui(self):
        """Setup the plugin panel UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Title
        title = QtWidgets.QLabel("Plugins")
        title.setStyleSheet("font-weight: bold; font-size: 10pt;")
        layout.addWidget(title)
        
        # Use QToolBox for collapsible sections
        self.tool_box = QtWidgets.QToolBox()
        self.tool_box.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tool_box)
        
        # --- Plugin List Section ---
        self.plugin_list_widget = QtWidgets.QWidget()
        plugin_list_layout = QtWidgets.QVBoxLayout(self.plugin_list_widget)
        plugin_list_layout.setContentsMargins(2, 2, 2, 2)
        plugin_list_layout.setSpacing(2)
        
        self.plugin_list = QtWidgets.QListWidget()
        self.plugin_list.setMaximumHeight(200)  # Limit height to save space
        self.plugin_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.plugin_list.itemClicked.connect(self._on_plugin_selected)
        self.plugin_list.itemChanged.connect(self._on_plugin_toggled)
        self.plugin_list.itemEntered.connect(self._on_plugin_hover)  # Show tooltip on hover
        plugin_list_layout.addWidget(self.plugin_list)
        
        self.tool_box.addItem(self.plugin_list_widget, "Plugin List")
        
        # --- Configuration Section ---
        self.config_widget = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout(self.config_widget)
        config_layout.setContentsMargins(2, 2, 2, 2)
        config_layout.setSpacing(2)
        
        self.config_text = QtWidgets.QTextEdit()
        self.config_text.setMaximumHeight(100)  # Reduced height
        self.config_text.setPlaceholderText("Plugin configuration (JSON format)")
        config_layout.addWidget(self.config_text)
        
        # Config buttons
        config_buttons = QtWidgets.QWidget()
        config_btn_layout = QtWidgets.QHBoxLayout(config_buttons)
        config_btn_layout.setContentsMargins(0, 0, 0, 0)
        config_btn_layout.setSpacing(2)
        
        self.load_config_btn = QtWidgets.QPushButton("Load")
        self.load_config_btn.setMaximumWidth(60)
        self.load_config_btn.clicked.connect(self._load_plugin_config)
        config_btn_layout.addWidget(self.load_config_btn)
        
        self.save_config_btn = QtWidgets.QPushButton("Save")
        self.save_config_btn.setMaximumWidth(60)
        self.save_config_btn.clicked.connect(self._save_plugin_config)
        config_btn_layout.addWidget(self.save_config_btn)
        
        config_layout.addWidget(config_buttons)
        self.tool_box.addItem(self.config_widget, "Config")
        
        # --- Actions Section ---
        self.actions_widget = QtWidgets.QWidget()
        actions_layout = QtWidgets.QVBoxLayout(self.actions_widget)
        actions_layout.setContentsMargins(2, 2, 2, 2)
        actions_layout.setSpacing(2)
        
        action_buttons = QtWidgets.QWidget()
        action_btn_layout = QtWidgets.QHBoxLayout(action_buttons)
        action_btn_layout.setContentsMargins(0, 0, 0, 0)
        action_btn_layout.setSpacing(2)
        
        self.run_plugin_btn = QtWidgets.QPushButton("Run")
        self.run_plugin_btn.clicked.connect(self._run_plugin_on_current_data)
        self.run_plugin_btn.setEnabled(False)
        action_btn_layout.addWidget(self.run_plugin_btn)
        
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_plugins)
        action_btn_layout.addWidget(self.refresh_btn)
        
        actions_layout.addWidget(action_buttons)
        self.tool_box.addItem(self.actions_widget, "Actions")
        
        # --- Results Section ---
        self.results_widget = QtWidgets.QWidget()
        results_layout = QtWidgets.QVBoxLayout(self.results_widget)
        results_layout.setContentsMargins(2, 2, 2, 2)
        
        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setMaximumHeight(60)  # Reduced height
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Results")
        results_layout.addWidget(self.results_text)
        
        self.tool_box.addItem(self.results_widget, "Results")
        
        # Set initial sizes - collapse all except Plugin List
        self.tool_box.setCurrentIndex(0)  # Start with Plugin List expanded
    
    def _load_plugin_manager(self):
        """Load the plugin manager."""
        try:
            from plugins.plugin_manager import get_plugin_manager
            self.plugin_manager = get_plugin_manager()
            
            # Set custom plugin directory
            self.custom_plugin_dir = Path.cwd() / 'plugin'
            if self.custom_plugin_dir.exists():
                self.plugin_manager.add_plugin_directory(self.custom_plugin_dir)
            
            # Refresh plugin list
            self._refresh_plugins()
            
        except Exception as e:
            self.results_text.setText(f"Error loading plugin manager: {e}")
    
    def _refresh_plugins(self):
        """Refresh the list of available plugins."""
        if self.plugin_manager is None:
            return
        
        self.plugin_list.clear()
        self.plugin_configs.clear()
        
        try:
            # Load plugins from custom directory
            if self.custom_plugin_dir and self.custom_plugin_dir.exists():
                for plugin_file in self.custom_plugin_dir.glob("*.py"):
                    if plugin_file.name.startswith("_"):
                        continue
                    
                    plugin_name = plugin_file.stem
                    try:
                        self.plugin_manager.load_plugin(plugin_name, plugin_file)
                    except Exception as e:
                        print(f"[PluginPanel] Failed to load {plugin_name}: {e}")
            
            # Get all loaded plugins
            all_plugins = self.plugin_manager.get_all_plugins()
            
            for plugin_name, plugin in all_plugins.items():
                # Create list item
                item = QtWidgets.QListWidgetItem(plugin_name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                
                # Check config file for enabled state
                config_file = self.custom_plugin_dir / f"{plugin_name}_config.json"
                is_enabled = True  # Default to enabled
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            config = json.load(f)
                            is_enabled = config.get("enabled", True)
                    except Exception:
                        pass
                
                item.setCheckState(Qt.CheckState.Checked if is_enabled else Qt.CheckState.Unchecked)
                
                # Store plugin reference
                item.setData(Qt.ItemDataRole.UserRole, plugin)
                
                self.plugin_list.addItem(item)
                
                # Load config if available
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            self.plugin_configs[plugin_name] = json.load(f)
                    except Exception:
                        pass
            
            self.results_text.setText(f"Loaded {len(all_plugins)} plugins")
            
        except Exception as e:
            self.results_text.setText(f"Error refreshing plugins: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_plugin_selected(self, item):
        """Handle plugin selection."""
        plugin = item.data(Qt.ItemDataRole.UserRole)
        plugin_name = item.text()
        
        if plugin is None:
            return
        
        # Show tooltip on selection as well
        try:
            status = "Enabled" if item.checkState() == Qt.CheckState.Checked else "Disabled"
            tooltip_text = f"<b>{plugin.get_name()}</b><br>"
            tooltip_text += f"<b>Version:</b> {plugin.get_version()}<br>"
            tooltip_text += f"<b>Author:</b> {getattr(plugin, 'author', 'Unknown')}<br>"
            tooltip_text += f"<b>Status:</b> {status}<br>"
            tooltip_text += f"<b>Description:</b> {plugin.get_description()}"
            
            QtWidgets.QToolTip.showText(
                self.plugin_list.mapToGlobal(self.plugin_list.visualItemRect(item).bottomLeft()),
                tooltip_text,
                self.plugin_list
            )
        except Exception as e:
            print(f"[PluginPanel] Error showing tooltip: {e}")
        
        # Load config into text area (details removed, use tooltips instead)
        if plugin_name in self.plugin_configs:
            config_json = json.dumps(self.plugin_configs[plugin_name], indent=2)
            self.config_text.setPlainText(config_json)
        else:
            self.config_text.clear()
        
        # Enable run button if plugin has manual execution
        self.run_plugin_btn.setEnabled(hasattr(plugin, 'manual_execute_with_data'))
    
    def _on_plugin_hover(self, item):
        """Show plugin details as tooltip when hovering over plugin item."""
        plugin = item.data(Qt.ItemDataRole.UserRole)
        if plugin is None:
            return
        
        try:
            status = "Enabled" if item.checkState() == Qt.CheckState.Checked else "Disabled"
            tooltip_text = f"<b>{plugin.get_name()}</b><br>"
            tooltip_text += f"<b>Version:</b> {plugin.get_version()}<br>"
            tooltip_text += f"<b>Author:</b> {getattr(plugin, 'author', 'Unknown')}<br>"
            tooltip_text += f"<b>Status:</b> {status}<br>"
            tooltip_text += f"<b>Description:</b> {plugin.get_description()}"
            
            QtWidgets.QToolTip.showText(
                self.plugin_list.mapToGlobal(self.plugin_list.visualItemRect(item).bottomLeft()),
                tooltip_text,
                self.plugin_list
            )
        except Exception as e:
            print(f"[PluginPanel] Error showing tooltip: {e}")
    
    def _on_plugin_toggled(self, item):
        """Handle plugin enable/disable toggle."""
        plugin = item.data(Qt.ItemDataRole.UserRole)
        if plugin is None:
            return
        
        is_enabled = item.checkState() == Qt.CheckState.Checked
        
        # Update plugin enabled state
        if hasattr(plugin, 'enabled'):
            plugin.enabled = is_enabled
        
        print(f"[PluginPanel] Plugin {item.text()} {'enabled' if is_enabled else 'disabled'}")
        
        print(f"[PluginPanel] Plugin {item.text()} {'enabled' if is_enabled else 'disabled'}")
    
    def _load_plugin_config(self):
        """Load configuration for selected plugin."""
        current_item = self.plugin_list.currentItem()
        if current_item is None:
            return
        
        plugin_name = current_item.text()
        config_text = self.config_text.toPlainText()
        
        if not config_text.strip():
            return
        
        try:
            config = json.loads(config_text)
            self.plugin_configs[plugin_name] = config
            
            # Also save to file
            config_file = self.custom_plugin_dir / f"{plugin_name}_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.results_text.setText(f"Configuration loaded for {plugin_name}")
            
        except json.JSONDecodeError as e:
            self.results_text.setText(f"Invalid JSON: {e}")
        except Exception as e:
            self.results_text.setText(f"Error loading config: {e}")
    
    def _save_plugin_config(self):
        """Save configuration for selected plugin."""
        current_item = self.plugin_list.currentItem()
        if current_item is None:
            return
        
        plugin_name = current_item.text()
        config_text = self.config_text.toPlainText()
        
        if not config_text.strip():
            return
        
        try:
            config = json.loads(config_text)
            self.plugin_configs[plugin_name] = config
            
            # Save to file
            config_file = self.custom_plugin_dir / f"{plugin_name}_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Re-initialize plugin with new config
            plugin = current_item.data(Qt.ItemDataRole.UserRole)
            if plugin and hasattr(plugin, 'initialize'):
                plugin.initialize(config)
            
            self.results_text.setText(f"Configuration saved for {plugin_name}")
            
        except json.JSONDecodeError as e:
            self.results_text.setText(f"Invalid JSON: {e}")
        except Exception as e:
            self.results_text.setText(f"Error saving config: {e}")
    
    def _run_plugin_on_current_data(self):
        """Run selected plugin on current data."""
        current_item = self.plugin_list.currentItem()
        if current_item is None:
            return
        
        plugin = current_item.data(Qt.ItemDataRole.UserRole)
        plugin_name = current_item.text()
        
        if not hasattr(plugin, 'manual_execute_with_data'):
            self.results_text.setText(f"Plugin {plugin_name} does not support manual execution")
            return
        
        try:
            # Get data from LiveTab
            if self.parent_window and hasattr(self.parent_window, 'live_tab'):
                scan_data = self.parent_window.live_tab.get_multiaxis_scan_data()
                
                if scan_data is None:
                    self.results_text.setText("No scan data available. Run a multi-axis scan first.")
                    return
                
                # Load config
                config = self.plugin_configs.get(plugin_name, {})
                plugin.initialize(config)
                
                # Execute plugin
                success = plugin.manual_execute_with_data(
                    scan_data['detector_data'],
                    scan_data['positions'],
                    scan_data.get('scan_dimensions')
                )
                
                if success:
                    self.results_text.setText(f"Plugin {plugin_name} executed successfully")
                    
                    # Register with detector image panel for tooltips
                    if hasattr(self.parent_window.live_tab, 'detector_image_panel'):
                        self.parent_window.live_tab.detector_image_panel.register_decoder_plugin(plugin_name, plugin)
                else:
                    self.results_text.setText(f"Plugin {plugin_name} execution failed")
            else:
                self.results_text.setText("LiveTab not available")
                
        except Exception as e:
            self.results_text.setText(f"Error running plugin: {e}")
            import traceback
            traceback.print_exc()
    
    def set_parent_window(self, parent):
        """Set the parent window for accessing other components."""
        self.parent_window = parent
