import pyvisa
from datetime import datetime

class Multimeter:
   def __init__(self, gpib = None,nplc = 0.02):
      rm = pyvisa.ResourceManager()
      if gpib is None:
            gpib = '11'
      dmm = rm.open_resource('GPIB0::'+ gpib + '::INSTR')  # Replace with your GPIB address

      dmm.write_termination = '\n'
      dmm.read_termination = '\n'
      dmm.timeout = 3000

      dmm.write("*RST")
      dmm.write("CONF:VOLT:DC")
      dmm.write(f"VOLT:DC:NPLC {nplc}")
      dmm.write("TRIG:SOUR IMM")
      dmm.write("SAMP:COUN 1")

      print("Multimeter ready:", dmm.query("*IDN?"))
      self.dmm = dmm
      self.name = dmm.query("*IDN?").strip()
      self.scale = 1.0
      self.offset = 0.0

   def read_voltage(self):
      if self.dmm is None:
            raise('multimeter is not initialized correctly!')
      
      # timestamp = datetime.now().isoformat(timespec='milliseconds')
      voltage = float(self.dmm.query("READ?").strip())
      return voltage
   
   def close(self):
      self.dmm.close()

   def set_scale(self, scale: float, offset: float):
      self.scale = float(scale)
      self.offset = float(offset)

   def read_value(self):
      raw = self.read_voltage()
      try:
         return float(self.scale) * float(raw) + float(self.offset)
      except Exception:
         return raw

   def disconnect(self):
      self.dmm.close()

