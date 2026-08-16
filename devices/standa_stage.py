from pylablib.devices import Standa
from .base import SingleAxis
import logging
import argparse


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
      
      try:
         self.dev.move_by(int(delta))
         
         # Try to wait for move completion with timeout
         try:
            self.dev.wait_move(timeout=30.0)
         except Exception as e:
            logger.warning(f"Wait move failed for {self.com_port}: {e}")
            # Continue anyway - move may have completed
         
         try:
            self.pos = self.dev.get_position()
         except Exception as e:
            logger.warning(f"Failed to get position after move: {e}")
            self.pos += int(delta)
      except Exception as e:
         logger.error(f"Failed to move axis {self.com_port}: {e}")
         raise

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
      
      # Move X and Y with error handling
      x_success = False
      y_success = False
      
      # Try to move X axis
      try:
         self.x.move_to(x)
         x_success = True
      except Exception as e:
         logger.warning(f"Failed to move X axis: {e}")
      
      # Try to move Y axis
      try:
         self.y.move_to(y)
         y_success = True
      except Exception as e:
         logger.warning(f"Failed to move Y axis: {e}")
      
      # If both failed, raise exception
      if not x_success and not y_success:
         raise RuntimeError("Failed to move both X and Y axes")
      
      # Return success status
      return x_success and y_success

   def move_by(self, dx: float, dy: float):
      if dx: self.x.move_by(dx)
      if dy: self.y.move_by(dy)

   def get_position(self):
      return self.x.get_position(), self.y.get_position()

   def stop(self):
      self.x.stop()
      self.y.stop()


def main():
   parser = argparse.ArgumentParser(description="Quick Standa stage tester")
   parser.add_argument("--com", required=True, help="COM port, e.g. COM7")
   parser.add_argument("--delta", type=float, default=0.0, help="Relative move in steps")
   parser.add_argument("--target", type=float, help="Absolute target position in steps")
   parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
   args = parser.parse_args()

   logging.basicConfig(
      level=getattr(logging, str(args.log_level).upper(), logging.INFO),
      format="%(asctime)s %(levelname)s %(name)s: %(message)s",
   )

   axis = StandaAxis(args.com)
   try:
      print(f"Connected to {args.com}")
      print(f"Current position: {axis.get_position()}")

      if args.target is not None:
         print(f"Moving to target: {args.target}")
         axis.move_to(args.target)
         print(f"Position after move_to: {axis.get_position()}")
      elif args.delta != 0:
         print(f"Moving by delta: {args.delta}")
         axis.move_by(args.delta)
         print(f"Position after move_by: {axis.get_position()}")
      else:
         print("No move requested. Use --delta or --target to command motion.")
   except Exception as exc:
      logger.exception("Stage test failed: %s", exc)
      raise
   finally:
      axis.stop()
      axis.disconnect()
      print("Stage stopped and disconnected")


if __name__ == "__main__":
   main()