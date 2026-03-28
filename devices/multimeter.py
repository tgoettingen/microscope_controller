from __future__ import annotations

from typing import Dict, Any

try:
   import pyvisa
except Exception:
   pyvisa = None

from .base import Detector


class Multimeter(Detector):
   def __init__(self, gpib: str | int | None = None, nplc: float = 0.02, name: str | None = None, auto_connect: bool = True):
      # Use a stable default name even before connect() runs.
      gpib_str = str(gpib) if gpib is not None else "11"
      nm = name if name is not None else f"GPIB{gpib_str}"
      Detector.__init__(self, nm)
      self.gpib = gpib_str
      self.nplc = float(nplc)
      self.dmm = None
      self.scale = 1.0
      self.offset = 0.0
      self.last_error: str | None = None
      if auto_connect:
         # Orchestrator.initialize() will call connect() again; connect() is idempotent.
         self.connect()

   # ---- Device/Detector interface ----
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

      # Basic DC voltage configuration
      dmm.write("*RST")
      dmm.write("CONF:VOLT:DC")
      dmm.write(f"VOLT:DC:NPLC {self.nplc}")
      dmm.write("TRIG:SOUR IMM")
      dmm.write("SAMP:COUN 1")

      self.dmm = dmm
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
      }

   def reset(self) -> None:
      # Best-effort reset; keep safe when disconnected.
      if self.dmm is None:
         return
      try:
         self.dmm.write("*RST")
         self.dmm.write("CONF:VOLT:DC")
         self.dmm.write(f"VOLT:DC:NPLC {self.nplc}")
         self.dmm.write("TRIG:SOUR IMM")
         self.dmm.write("SAMP:COUN 1")
      except Exception:
         pass

   def set_scale(self, scale: float, offset: float = 0.0) -> None:
      self.scale = float(scale)
      self.offset = float(offset)

   def read_voltage(self) -> float:
      if self.dmm is None:
         raise RuntimeError("multimeter is not connected")
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

