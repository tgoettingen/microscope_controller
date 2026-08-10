"""Excitation Source Device Control using SerialLink protocol.

This module provides a device class for controlling excitation sources (LEDs, lasers, etc.)
using the SerialLink protocol from SerialLink.py. It can be integrated into multi-axis scans
for synchronized excitation control.
"""

import logging
from typing import Dict, Any, Optional
import queue
import threading
import time

try:
    from .SerialLink import SerialLink, SimLink
except ImportError:
    SerialLink = None
    SimLink = None

from .base import ExcitationSource

logger = logging.getLogger(__name__)


class ExcitationDevice(ExcitationSource):
    """Excitation source control using SerialLink protocol.
    
    This class provides ON/OFF control for excitation sources like LEDs or lasers
    using the MSP432-based SerialLink protocol. It can be integrated into multi-axis
    scans for synchronized excitation during measurements.
    
    Usage:
        # Create device
        excitation = ExcitationDevice("Excitation", port="COM3", channel=0, simulate=False)
        
        # Connect and control
        excitation.connect()
        excitation.on()  # Turn on excitation
        excitation.off()  # Turn off excitation
        excitation.set_channel(1)  # Switch to channel 1
        
        # Disconnect
        excitation.disconnect()
    """
    
    def __init__(self, name: str, port: str = None, channel: int = 0, simulate: bool = False):
        super().__init__(name)
        self._port = port
        self._channel = channel
        self._simulate = simulate
        self._rx_queue = queue.Queue()
        self._link = None
        self._current_channel = channel
        self._is_on = False
        
        # Thread-safe state
        self._lock = threading.Lock()
        
    def connect(self) -> None:
        """Connect to the excitation controller."""
        try:
            if self._simulate:
                self._link = SimLink(self._rx_queue)
                self._link.open("SIMULATED")
            else:
                if SerialLink is None:
                    raise RuntimeError("SerialLink not available")
                self._link = SerialLink(self._rx_queue)
                if self._port:
                    self._link.open(self._port)
                else:
                    raise RuntimeError("No serial port specified")
            
            self.connected = True
            logger.info("Excitation device %s connected to %s (simulate=%s)", 
                       self.name, self._port if not self._simulate else "SIMULATED", self._simulate)
            
            # Set initial channel
            self._set_channel_safe(self._channel)
            
        except Exception as e:
            logger.error("Failed to connect excitation device %s: %s", self.name, e)
            raise
    
    def disconnect(self) -> None:
        """Disconnect from the excitation controller."""
        try:
            with self._lock:
                if self._link and self._link.is_open:
                    self._link.close()
                self._link = None
                self.connected = False
                self._is_on = False
            logger.info("Excitation device %s disconnected", self.name)
        except Exception as e:
            logger.error("Error disconnecting excitation device %s: %s", self.name, e)
    
    def reset(self) -> None:
        """Reset the excitation source to OFF state."""
        try:
            self.off()
            logger.info("Excitation device %s reset", self.name)
        except Exception as e:
            logger.error("Error resetting excitation device %s: %s", self.name, e)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return device capabilities."""
        return {
            "kind": "excitation_source",
            "simulate": self._simulate,
            "channels": list(range(8)) if self._simulate else ["LED1", "LED_R", "LED_G", "LED_B", "TTL0", "TTL1", "TTL2", "TTL3"],
            "supports_on_off": True,
            "supports_channel_selection": True,
            "supports_set": True
        }
    
    def on(self) -> None:
        """Turn on the excitation source."""
        with self._lock:
            # Auto-connect if not connected (for real devices only)
            if not self.connected and not self._simulate:
                try:
                    self.connect()
                    logger.info("Auto-connected excitation device %s", self.name)
                except Exception as e:
                    logger.error("Failed to auto-connect excitation device %s: %s", self.name, e)
                    raise
            elif not self.connected and self._simulate:
                # Simulated devices are always "connected"
                self.connected = True
            
            try:
                self._send_command(f"SET {self._current_channel} 1")
                self._is_on = True
                logger.debug("Excitation device %s turned ON (channel %d)", self.name, self._current_channel)
            except Exception as e:
                logger.error("Error turning on excitation device %s: %s", self.name, e)
                raise
    
    def off(self) -> None:
        """Turn off the excitation source."""
        with self._lock:
            # Auto-connect if not connected (only for real devices)
            if not self.connected and not self._simulate:
                try:
                    self.connect()
                    logger.info("Auto-connected excitation device %s", self.name)
                except Exception as e:
                    logger.error("Failed to auto-connect excitation device %s: %s", self.name, e)
                    raise
            
            try:
                self._send_command(f"SET {self._current_channel} 0")
                self._is_on = False
                logger.debug("Excitation device %s turned OFF (channel %d)", self.name, self._current_channel)
            except Exception as e:
                logger.error("Error turning off excitation device %s: %s", self.name, e)
                raise
    
    def set_channel(self, channel: int) -> None:
        """Set the excitation channel.
        
        Args:
            channel: Channel number (0-7 for MSP432 controller)
        """
        with self._lock:
            # Auto-connect if not connected (only for real devices)
            if not self.connected and not self._simulate:
                try:
                    self.connect()
                    logger.info("Auto-connected excitation device %s", self.name)
                except Exception as e:
                    logger.error("Failed to auto-connect excitation device %s: %s", self.name, e)
                    raise
            
            self._set_channel_safe(channel)
    
    def _set_channel_safe(self, channel: int) -> None:
        """Internal method to set channel with proper error handling."""
        try:
            self._send_command(f"SET {channel} {1 if self._is_on else 0}")
            self._current_channel = channel
            logger.debug("Excitation device %s set to channel %d", self.name, channel)
        except Exception as e:
            logger.error("Error setting channel on excitation device %s: %s", self.name, e)
            raise
    
    def get_channel(self) -> int:
        """Get the current excitation channel."""
        with self._lock:
            return self._current_channel
    
    def _send_command(self, command: str) -> None:
        """Send a command to the SerialLink and wait for response."""
        if not self._link or not self._link.is_open:
            raise RuntimeError("Device not connected")
        
        try:
            self._link.send(command)
            # Small delay to allow command processing
            time.sleep(0.05)
        except Exception as e:
            logger.error("Error sending command '%s' to excitation device %s: %s", 
                         command, self.name, e)
            raise
    
    def toggle(self) -> None:
        """Toggle the excitation source on/off state."""
        with self._lock:
            if self._is_on:
                self.off()
            else:
                self.on()
    
    def is_on(self) -> bool:
        """Check if the excitation source is currently on."""
        with self._lock:
            return self._is_on
    
    def all_off(self) -> None:
        """Turn off all excitation channels."""
        try:
            self._send_command("ALLOFF")
            with self._lock:
                self._is_on = False
            logger.info("Excitation device %s all channels turned OFF", self.name)
        except Exception as e:
            logger.error("Error turning off all channels on excitation device %s: %s", self.name, e)
            raise


# Simulated version for testing without hardware
class SimulatedExcitationDevice(ExcitationSource):
    """Simulated excitation source for testing without hardware."""
    
    def __init__(self, name: str = "SimulatedExcitation"):
        super().__init__(name)
        self._current_channel = 0
        self._is_on = False
        self._channels = ["LED1", "LED_R", "LED_G", "LED_B", "TTL0", "TTL1", "TTL2", "TTL3"]
    
    def connect(self) -> None:
        """Simulate connection."""
        self.connected = True
        logger.info("Simulated excitation device %s connected", self.name)
    
    def disconnect(self) -> None:
        """Simulate disconnection."""
        self.connected = False
        self._is_on = False
        logger.info("Simulated excitation device %s disconnected", self.name)
    
    def reset(self) -> None:
        """Reset to OFF state."""
        self._is_on = False
        logger.info("Simulated excitation device %s reset", self.name)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return simulated capabilities."""
        return {
            "kind": "excitation_source",
            "simulate": True,
            "channels": self._channels,
            "supports_on_off": True,
            "supports_channel_selection": True,
            "supports_set": True
        }
    
    def on(self) -> None:
        """Simulate turning on."""
        # Auto-connect if not connected
        if not self.connected:
            self.connect()
        
        self._is_on = True
        logger.debug("Simulated excitation device %s turned ON (channel %s)", 
                    self.name, self._channels[self._current_channel])
    
    def off(self) -> None:
        """Simulate turning off."""
        # Auto-connect if not connected
        if not self.connected:
            self.connect()
        
        self._is_on = False
        logger.debug("Simulated excitation device %s turned OFF (channel %s)", 
                    self.name, self._channels[self._current_channel])
    
    def set_channel(self, channel: int) -> None:
        """Simulate channel change."""
        # Auto-connect if not connected
        if not self.connected:
            self.connect()
        
        if 0 <= channel < len(self._channels):
            self._current_channel = channel
            logger.debug("Simulated excitation device %s set to channel %s (%s)", 
                        self.name, channel, self._channels[channel])
        else:
            raise ValueError(f"Invalid channel {channel}")
    
    def get_channel(self) -> int:
        """Get current channel."""
        return self._current_channel
    
    def toggle(self) -> None:
        """Toggle on/off state."""
        if self._is_on:
            self.off()
        else:
            self.on()
    
    def is_on(self) -> bool:
        """Check if on."""
        return self._is_on
    
    def all_off(self) -> None:
        """Turn off all channels."""
        self._is_on = False
        logger.info("Simulated excitation device %s all channels turned OFF", self.name)