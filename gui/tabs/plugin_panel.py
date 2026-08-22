"""
Plugin Panel - GUI for managing microscope controller plugins.

This panel provides:
- List of available plugins
- Enable/disable plugins
- Plugin configuration
- Apply plugins on selection
- View plugin status
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
        
        # Use a scroll area to allow both sections to be visible
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setContentsMargins(0, 0, 0, 0)
        
        # Create container widget for scroll area
        container_widget = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container_widget)
        container_layout.setContentsMargins(2, 2, 2, 2)
        container_layout.setSpacing(4)
        
        # --- Plugin List Section ---
        self.plugin_list_group = QtWidgets.QGroupBox("Plugin List")
        self.plugin_list_group.setCheckable(True)
        self.plugin_list_group.setChecked(True)
        self.plugin_list_group.toggled.connect(self._toggle_plugin_list)
        plugin_list_layout = QtWidgets.QVBoxLayout(self.plugin_list_group)
        plugin_list_layout.setContentsMargins(2, 2, 2, 2)
        plugin_list_layout.setSpacing(2)
        
        # Create a container widget for the plugin list content
        self.plugin_list_content = QtWidgets.QWidget()
        plugin_list_content_layout = QtWidgets.QVBoxLayout(self.plugin_list_content)
        plugin_list_content_layout.setContentsMargins(0, 0, 0, 0)
        plugin_list_content_layout.setSpacing(2)
        
        self.plugin_list = QtWidgets.QListWidget()
        self.plugin_list.setMaximumHeight(150)  # Limit height to save space
        self.plugin_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.plugin_list.itemClicked.connect(self._on_plugin_clicked)
        self.plugin_list.itemChanged.connect(self._on_plugin_toggled)
        self.plugin_list.itemEntered.connect(self._on_plugin_hover)  # Show tooltip on hover
        plugin_list_content_layout.addWidget(self.plugin_list)
        
        # Add refresh button below plugin list
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_plugins)
        plugin_list_content_layout.addWidget(refresh_btn)
        
        plugin_list_layout.addWidget(self.plugin_list_content)
        container_layout.addWidget(self.plugin_list_group)
        
        # --- Configuration Section ---
        self.config_group = QtWidgets.QGroupBox("Config")
        self.config_group.setCheckable(True)
        self.config_group.setChecked(True)
        self.config_group.toggled.connect(self._toggle_config)
        config_layout = QtWidgets.QVBoxLayout(self.config_group)
        config_layout.setContentsMargins(2, 2, 2, 2)
        config_layout.setSpacing(2)
        
        # Create a container widget for the config content
        self.config_content = QtWidgets.QWidget()
        config_content_layout = QtWidgets.QVBoxLayout(self.config_content)
        config_content_layout.setContentsMargins(0, 0, 0, 0)
        config_content_layout.setSpacing(2)
        
        self.config_text = QtWidgets.QTextEdit()
        self.config_text.setMaximumHeight(100)  # Reduced height
        self.config_text.setPlaceholderText("Plugin configuration (JSON format)")
        config_content_layout.addWidget(self.config_text)
        
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
        
        config_content_layout.addWidget(config_buttons)
        config_layout.addWidget(self.config_content)
        container_layout.addWidget(self.config_group)
        
        # Add stretch to push everything to top
        container_layout.addStretch()
        
        scroll_area.setWidget(container_widget)
        layout.addWidget(scroll_area)
    
    def _toggle_plugin_list(self, checked):
        """Toggle plugin list visibility."""
        if hasattr(self, 'plugin_list_content'):
            self.plugin_list_content.setVisible(checked)
    
    def _toggle_config(self, checked):
        """Toggle config section visibility."""
        if hasattr(self, 'config_content'):
            self.config_content.setVisible(checked)
    
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
            if self.parent_window:
                try:
                    self.parent_window.statusBar().showMessage(f"Error loading plugin manager: {e}", 5000)
                except Exception:
                    pass
    
    def _refresh_plugins(self):
        """Refresh the list of available plugins."""
        if self.plugin_manager is None:
            return
        
        self.plugin_list.clear()
        self.plugin_configs.clear()
        
        try:
            # Load configs from individual config files first
            if self.custom_plugin_dir and self.custom_plugin_dir.exists():
                for config_file in self.custom_plugin_dir.glob("*_config.json"):
                    plugin_name = config_file.stem.replace("_config", "")
                    try:
                        with open(config_file) as f:
                            self.plugin_configs[plugin_name] = json.load(f)
                    except Exception as e:
                        print(f"[PluginPanel] Error loading config for {plugin_name}: {e}")
            
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
                is_enabled = False  # Default to disabled
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            config = json.load(f)
                            is_enabled = config.get("enabled", False)
                            print(f"[PluginPanel] Config for {plugin_name}: enabled={is_enabled}")
                    except Exception as e:
                        print(f"[PluginPanel] Error reading config for {plugin_name}: {e}")
                else:
                    print(f"[PluginPanel] No config file for {plugin_name}, defaulting to disabled")
                
                item.setCheckState(Qt.CheckState.Checked if is_enabled else Qt.CheckState.Unchecked)
                
                # Store plugin reference
                item.setData(Qt.ItemDataRole.UserRole, plugin)
                
                # Sync plugin's enabled state with config
                if hasattr(plugin, 'enabled'):
                    plugin.enabled = is_enabled
                
                self.plugin_list.addItem(item)
                
                # Load config if available
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            self.plugin_configs[plugin_name] = json.load(f)
                    except Exception:
                        pass
            
            if self.parent_window:
                try:
                    self.parent_window.statusBar().showMessage(f"Loaded {len(all_plugins)} plugins", 5000)
                except Exception:
                    pass
            
        except Exception as e:
            if self.parent_window:
                try:
                    self.parent_window.statusBar().showMessage(f"Error refreshing plugins: {e}", 5000)
                except Exception:
                    pass
            import traceback
            traceback.print_exc()
    
    def _on_plugin_clicked(self, item):
        """Handle plugin item click - apply plugin immediately."""
        print(f"[PluginPanel] _on_plugin_clicked called for {item.text()}")
        
        # Call the selection handler first to load config
        self._on_plugin_selected(item)
        
        # Run the plugin immediately
        self._run_plugin_on_current_data()
    
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
        print("[PluginPanel] _on_plugin_toggled called!")
        plugin = item.data(Qt.ItemDataRole.UserRole)
        if plugin is None:
            print("[PluginPanel] Plugin is None, returning")
            return
        
        is_enabled = item.checkState() == Qt.CheckState.Checked
        print(f"[PluginPanel] Plugin {item.text()} toggled to {'enabled' if is_enabled else 'disabled'}")
        
        # Update config file to persist the enabled state
        plugin_name = item.text()
        config_file = self.custom_plugin_dir / f"{plugin_name}_config.json"
        print(f"[PluginPanel] Config file path: {config_file}")
        print(f"[PluginPanel] Config file exists: {config_file.exists()}")
        
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                print(f"[PluginPanel] Config before update: {config}")
                config["enabled"] = is_enabled
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"[PluginPanel] Updated config file {config_file}: enabled={is_enabled}")
                print(f"[PluginPanel] Config after update: {config}")
            except Exception as e:
                print(f"[PluginPanel] Error updating config file: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[PluginPanel] Config file does not exist, creating new one")
            try:
                config = {"enabled": is_enabled}
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"[PluginPanel] Created new config file {config_file}: enabled={is_enabled}")
            except Exception as e:
                print(f"[PluginPanel] Error creating config file: {e}")
                import traceback
                traceback.print_exc()
        
        # Update the plugin instance's enabled state in the plugin manager
        if self.plugin_manager:
            try:
                if is_enabled:
                    self.plugin_manager.enable_plugin(plugin_name)
                    print(f"[PluginPanel] Enabled plugin in plugin manager: {plugin_name}")
                else:
                    self.plugin_manager.disable_plugin(plugin_name)
                    print(f"[PluginPanel] Disabled plugin in plugin manager: {plugin_name}")
            except Exception as e:
                print(f"[PluginPanel] Error updating plugin manager: {e}")
                import traceback
                traceback.print_exc()
        
        # Update the plugin instance directly if available
        if plugin:
            try:
                plugin.enabled = is_enabled
                print(f"[PluginPanel] Updated plugin instance enabled state: {plugin_name} -> {is_enabled}")
            except Exception as e:
                print(f"[PluginPanel] Error updating plugin instance: {e}")
                import traceback
                traceback.print_exc()
        
        # Update the internal config storage
        if plugin_name in self.plugin_configs:
            self.plugin_configs[plugin_name]["enabled"] = is_enabled
            print(f"[PluginPanel] Updated internal config storage for {plugin_name}: enabled={is_enabled}")
        else:
            self.plugin_configs[plugin_name] = {"enabled": is_enabled}
            print(f"[PluginPanel] Created new config entry for {plugin_name}: enabled={is_enabled}")
        
        # Refresh the config display if this is the currently selected plugin
        current_item = self.plugin_list.currentItem()
        if current_item and current_item.text() == plugin_name:
            print(f"[PluginPanel] Refreshing config display for selected plugin: {plugin_name}")
            # Update the config text display with the updated config
            config_json = json.dumps(self.plugin_configs[plugin_name], indent=2)
            self.config_text.setPlainText(config_json)
            print(f"[PluginPanel] Updated config text display with: {config_json}")
    
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
            
            if self.parent_window:
                try:
                    self.parent_window.statusBar().showMessage(f"Configuration loaded for {plugin_name}", 5000)
                except Exception:
                    pass
            
        except json.JSONDecodeError as e:
            if self.parent_window:
                try:
                    self.parent_window.statusBar().showMessage(f"Invalid JSON: {e}", 5000)
                except Exception:
                    pass
        except Exception as e:
            if self.parent_window:
                try:
                    self.parent_window.statusBar().showMessage(f"Error loading config: {e}", 5000)
                except Exception:
                    pass
    
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
            
            if self.parent_window:
                try:
                    self.parent_window.statusBar().showMessage(f"Configuration saved for {plugin_name}", 5000)
                except Exception:
                    pass
            
        except json.JSONDecodeError as e:
            if self.parent_window:
                try:
                    self.parent_window.statusBar().showMessage(f"Invalid JSON: {e}", 5000)
                except Exception:
                    pass
        except Exception as e:
            if self.parent_window:
                try:
                    self.parent_window.statusBar().showMessage(f"Error saving config: {e}", 5000)
                except Exception:
                    pass
    
    def _run_plugin_on_current_data(self):
        """Run selected plugin on current data."""
        current_item = self.plugin_list.currentItem()
        if current_item is None:
            return
        
        plugin = current_item.data(Qt.ItemDataRole.UserRole)
        plugin_name = current_item.text()
        
        if not hasattr(plugin, 'manual_execute_with_data'):
            if self.parent_window:
                try:
                    self.parent_window.statusBar().showMessage(f"Plugin {plugin_name} does not support manual execution", 5000)
                except Exception:
                    pass
            return
        
        try:
            # Get data from LiveTab
            if self.parent_window and hasattr(self.parent_window, 'live_tab'):
                scan_data = self.parent_window.live_tab.get_multiaxis_scan_data()
                
                if scan_data is None:
                    if self.parent_window:
                        try:
                            self.parent_window.statusBar().showMessage("No scan data available. Run a multi-axis scan first.", 5000)
                        except Exception:
                            pass
                    return
                
                # Load config
                config = self.plugin_configs.get(plugin_name, {})
                
                # Add stage range from hardware config if available
                try:
                    if self.parent_window and hasattr(self.parent_window, 'stage_control_tab') and self.parent_window.stage_control_tab:
                        stage_config = getattr(self.parent_window.stage_control_tab, 'stage_config', {})
                        stage_range = {
                            'stage_x_min': stage_config.get('x_min'),
                            'stage_x_max': stage_config.get('x_max'),
                            'stage_y_min': stage_config.get('y_min'),
                            'stage_y_max': stage_config.get('y_max'),
                        }
                        config.update(stage_range)
                        print(f"[PluginPanel] Added stage range to config: {stage_range}")
                except Exception as e:
                    print(f"[PluginPanel] Error getting stage range: {e}")
                
                plugin.initialize(config)
                
                # Execute plugin
                success = plugin.manual_execute_with_data(
                    scan_data['detector_data'],
                    scan_data['positions'],
                    scan_data.get('scan_dimensions')
                )
                
                if success:
                    if self.parent_window:
                        try:
                            self.parent_window.statusBar().showMessage(f"Plugin {plugin_name} executed successfully", 5000)
                        except Exception:
                            pass
                    
                    # Register with detector image panel for tooltips
                    if hasattr(self.parent_window.live_tab, 'detector_image_panel'):
                        self.parent_window.live_tab.detector_image_panel.register_decoder_plugin(plugin_name, plugin)
                else:
                    if self.parent_window:
                        try:
                            self.parent_window.statusBar().showMessage(f"Plugin {plugin_name} execution failed", 5000)
                        except Exception:
                            pass
            else:
                if self.parent_window:
                    try:
                        self.parent_window.statusBar().showMessage("LiveTab not available", 5000)
                    except Exception:
                        pass
                
        except Exception as e:
            if self.parent_window:
                try:
                    self.parent_window.statusBar().showMessage(f"Error running plugin: {e}", 5000)
                except Exception:
                    pass
            import traceback
            traceback.print_exc()
    
    def set_parent_window(self, parent):
        """Set the parent window for accessing other components."""
        self.parent_window = parent
