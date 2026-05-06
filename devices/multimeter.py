from __future__ import annotations

from typing import Dict, Any

try:
   import pyvisa
except Exception:
   pyvisa = None

from .base import Detector


class Multimeter(Detector):
   def __init__(
      self,
      gpib: str | int | None = None,
      nplc: float = 0.02,
      name: str | None = None,
      auto_connect: bool = True,
      mode: str = "volt_dc",
   ):
      # Use a stable default name even before connect() runs.
      gpib_str = str(gpib) if gpib is not None else "11"
      nm = name if name is not None else f"GPIB{gpib_str}"
      Detector.__init__(self, nm)
      self.gpib = gpib_str
      self.nplc = float(nplc)
      self.mode = self._normalize_mode(mode)
      self.dmm = None
      self.scale = 1.0
      self.offset = 0.0
      self.last_error: str | None = None
      if auto_connect:
         # Orchestrator.initialize() will call connect() again; connect() is idempotent.
         self.connect()

   # ---- Device/Detector interface ----
   def _normalize_mode(self, mode: str) -> str:
      m = str(mode).strip().lower()
      aliases = {
         "volt": "volt_dc",
         "v": "volt_dc",
         "voltage": "volt_dc",
         "volt_dc": "volt_dc",
         "res": "res",
         "ohm": "res",
         "ohms": "res",
         "resistance": "res",
      }
      if m not in aliases:
         raise ValueError(f"Unsupported multimeter mode: {mode}")
      return aliases[m]

   def _apply_mode_config(self) -> None:
      if self.dmm is None:
         return
      if self.mode == "res":
         self.dmm.write("CONF:RES")
      else:
         self.dmm.write("CONF:VOLT:DC")
         self.dmm.write(f"VOLT:DC:NPLC {self.nplc}")
      self.dmm.write("TRIG:SOUR IMM")
      self.dmm.write("SAMP:COUN 1")

   def connect(self) -> None:
      if self.connected and self.dmm is not None:
         return
      if pyvisa is None:
         self.last_error = "pyvisa not available; install pyvisa (and a VISA backend)"
         self.connected = False
         self.dmm = None
         return

      try:
         rm = pyvisa.ResourceManager()
         dmm = rm.open_resource(f"GPIB0::{self.gpib}::INSTR")
      except Exception as e:
         self.last_error = f"Failed to open VISA resource for GPIB {self.gpib}: {e}"
         self.connected = False
         self.dmm = None
         return
      dmm.write_termination = "\n"
      dmm.read_termination = "\n"
      dmm.timeout = 3000

      # Configure current measurement mode.
      dmm.write("*RST")
      self.dmm = dmm
      self._apply_mode_config()

      try:
         self.name = dmm.query("*IDN?").strip() or self.name
      except Exception:
         pass
      self.last_error = None
      self.connected = True

   def disconnect(self) -> None:
      if self.dmm is not None:
         try:
            self.dmm.close()
         except Exception:
            pass
      self.dmm = None
      self.connected = False

   def get_capabilities(self) -> Dict[str, Any]:
      return {
         "type": "multimeter",
         "backend": "pyvisa" if pyvisa is not None else None,
         "gpib": self.gpib,
         "nplc": self.nplc,
         "mode": self.mode,
         "supported_modes": ["volt_dc", "res"],
      }

   def reset(self) -> None:
      # Best-effort reset; keep safe when disconnected.
      if self.dmm is None:
         return
      try:
         self.dmm.write("*RST")
         self._apply_mode_config()
      except Exception:
         pass

   def set_mode(self, mode: str) -> None:
      self.mode = self._normalize_mode(mode)
      if self.dmm is not None:
         self._apply_mode_config()

   def set_scale(self, scale: float, offset: float = 0.0) -> None:
      self.scale = float(scale)
      self.offset = float(offset)

   def read_voltage(self) -> float:
      if self.dmm is None:
         raise RuntimeError("multimeter is not connected")
      if self.mode != "volt_dc":
         self.set_mode("volt_dc")
      return float(self.dmm.query("READ?").strip())

   def read_resistance(self) -> float:
      if self.dmm is None:
         raise RuntimeError("multimeter is not connected")
      if self.mode != "res":
         self.set_mode("res")
      return float(self.dmm.query("READ?").strip())

   def read_value(self) -> float:
      raw = self.read_voltage()
      try:
         return float(self.scale) * float(raw) + float(self.offset)
      except Exception:
         return float(raw)

   # Backwards-compatible aliases
   def close(self) -> None:
      self.disconnect()

