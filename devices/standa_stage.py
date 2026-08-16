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
            # Handle specific timeout exceptions
            error_msg = str(e)
            if 'timeout' in error_msg.lower() or 'SerialTimeoutException' in error_msg:
               logger.warning(f"Move timeout for {self.com_port} - command may have completed despite timeout")
            else:
               logger.warning(f"Wait move failed for {self.com_port}: {e}")
            # Continue anyway - move may have completed
         
         try:
            self.pos = self.dev.get_position()
         except Exception as e:
            logger.warning(f"Failed to get position after move: {e}")
            self.pos += int(delta)
      except Exception as e:
         error_msg = str(e)
         # Handle specific timeout exceptions
         if 'timeout' in error_msg.lower() or 'SerialTimeoutException' in error_msg:
            logger.warning(f"Move command timeout for {self.com_port} - command may have been sent")
            # Still update position as best effort
            self.pos += int(delta)
         else:
            logger.error(f"Failed to move axis {self.com_port}: {e}")
            raise

   def move_to(self, target: float):
      try:
         logger.info("StandaAxis move_to (com=%s target=%s, current_pos=%s)", 
                    getattr(self, "com_port", None), target, self.pos)
      except Exception:
         pass
      # Like simulated stage, directly set the target position
      # and let the hardware handle the relative movement
      cur = self.get_position()
      delta = target - cur
      logger.info(f"StandaAxis move_to: target={target}, current={cur}, delta={delta}")
      self.move_by(delta)
      # After move, update position to target (like simulated stage)
      self.pos = target
      logger.info(f"StandaAxis move_to: new position={self.pos}")

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
      
      # Move X and Y simultaneously using separate threads for speed
      import threading
      x_success = False
      y_success = False
      x_error = None
      y_error = None
      
      # Create threads for concurrent movement
      x_thread = threading.Thread(
         target=self._move_x_axis,
         args=(x,),
         daemon=True
      )
      y_thread = threading.Thread(
         target=self._move_y_axis,
         args=(y,),
         daemon=True
      )
      
      # Start both threads simultaneously
      x_thread.start()
      y_thread.start()
      
      # Wait for both threads to complete
      x_thread.join(timeout=35.0)  # 30s timeout + 5s buffer
      y_thread.join(timeout=35.0)
      
      # Check thread results
      x_success = getattr(x_thread, 'success', False)
      y_success = getattr(y_thread, 'success', False)
      x_error = getattr(x_thread, 'error', None)
      y_error = getattr(y_thread, 'error', None)
      
      # Log results
      if x_success:
         logger.info("X axis move completed successfully")
      elif x_error:
         logger.warning(f"X axis move issue: {x_error}")
         
      if y_success:
         logger.info("Y axis move completed successfully")
      elif y_error:
         logger.warning(f"Y axis move issue: {y_error}")
      
      # If both failed with non-timeout errors, raise exception
      if not x_success and not y_success:
         raise RuntimeError(f"Failed to move both X and Y axes. X error: {x_error}, Y error: {y_error}")
      
      # Return success status
      return x_success and y_success
   
   def _move_x_axis(self, x: float):
      """Move X axis in a separate thread."""
      try:
         logger.info(f"StandaStageXY _move_x_axis: Moving X to {x}, current X position: {self.x.get_position()}")
         self.x.move_to(x)
         # Store success in thread object
         import threading
         threading.current_thread().success = True
         logger.info(f"StandaStageXY _move_x_axis: X move completed, new position: {self.x.get_position()}")
      except Exception as e:
         error_msg = str(e)
         # Handle timeout exceptions as warnings, not errors
         if 'timeout' in error_msg.lower() or 'SerialTimeoutException' in error_msg:
            logger.warning(f"X axis move timeout (may have completed): {e}")
            import threading
            threading.current_thread().success = True  # Consider as success for timeout cases
         else:
            logger.warning(f"Failed to move X axis: {e}")
            import threading
            threading.current_thread().success = False
            threading.current_thread().error = error_msg
   
   def _move_y_axis(self, y: float):
      """Move Y axis in a separate thread."""
      try:
         self.y.move_to(y)
         # Store success in thread object
         import threading
         threading.current_thread().success = True
      except Exception as e:
         error_msg = str(e)
         # Handle timeout exceptions as warnings, not errors
         if 'timeout' in error_msg.lower() or 'SerialTimeoutException' in error_msg:
            logger.warning(f"Y axis move timeout (may have completed): {e}")
            import threading
            threading.current_thread().success = True  # Consider as success for timeout cases
         else:
            logger.warning(f"Failed to move Y axis: {e}")
            import threading
            threading.current_thread().success = False
            threading.current_thread().error = error_msg

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