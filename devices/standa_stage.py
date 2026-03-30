from pylablib.devices import Standa
from .base import SingleAxis
import logging


logger = logging.getLogger(__name__)

class StandaAxis(SingleAxis):
   """Low-level controller for a single Standa axis on one COM port."""

   def __init__(self, com_port: str):
      self.com_port = com_port
      self.dev = None
      self.pos = 0

      try:
            self.dev = Standa.Standa8SMC(com_port)
      except Exception as e:
         try:
            logger.warning("Could not open Standa axis on %s: %s", com_port, e)
         except Exception:
            pass
            self.dev = None

   def connect(self):
      self.__init__()
   
   def disconnect(self):
      self.stop()
      
   def reset(self):
      pass
   
   
   def get_capabilities(self):
      pass

   def move_by(self, delta: float):
      try:
         logger.info("StandaAxis move_by (com=%s delta=%s)", getattr(self, "com_port", None), delta)
      except Exception:
         pass
      if self.dev is None:
         if self.com_port is not None:
            try:
               self.__init__(self.com_port)
            except:
               raise RuntimeError("Axis not available")
      self.dev.move_by(int(delta))
      self.dev.wait_move(timeout=30.0)
      try:
            self.pos = self.dev.get_position()
      except Exception:
            self.pos += int(delta)

   def move_to(self, target: float):
      try:
         logger.info("StandaAxis move_to (com=%s target=%s)", getattr(self, "com_port", None), target)
      except Exception:
         pass
      cur = self.get_position()
      delta = target - cur
      self.move_by(delta)

   def get_position(self) -> float:
      if self.dev is None:
            return self.pos
      try:
            self.pos = self.dev.get_position()
      except Exception:
            pass
      return self.pos

   def stop(self):
      if self.dev:
            try:
               self.dev.stop(immediate=True)
            except Exception:
               pass


class StandaStageXY:
   def __init__(self, com_x: str, com_y: str):
      self.x = StandaAxis(com_x)
      self.y = StandaAxis(com_y)

   def connect(self):
      pass
   
   def disconnect(self):
      self.stop()
      
   def reset(self):
      pass
   def get_capabilities(self):
      pass


   def move_to(self, x: float, y: float):
      try:
         logger.info("StandaStageXY move_to x=%s y=%s", x, y)
      except Exception:
         pass
      self.x.move_to(x)
      self.y.move_to(y)

   def move_by(self, dx: float, dy: float):
      if dx: self.x.move_by(dx)
      if dy: self.y.move_by(dy)

   def get_position(self):
      return self.x.get_position(), self.y.get_position()

   def stop(self):
      self.x.stop()
      self.y.stop()