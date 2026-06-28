import serial
import serial.tools.list_ports
import threading
import time
import os
import sys
from datetime import datetime


class UartReader:
    VREF = 2.048
    GAIN = 501
    ADC_MAX = 2 ** 23

    def __init__(self, port=None, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.valid_count = 0
        self.error_count = 0
        self.voltage = 0.0
        self.last_raw = b''
        self.last_adc = 0
        self.lock = threading.Lock()
        self._callbacks = []
        self._buffer = bytearray()

    @staticmethod
    def find_ch340_ports():
        ports = serial.tools.list_ports.comports()
        ch340 = []
        others = []
        for p in ports:
            desc = (p.description or '').upper()
            mfr = (p.manufacturer or '').upper()
            hwid = (p.hwid or '').upper()
            if 'CH340' in desc or 'CH340' in mfr or 'CH340' in hwid:
                ch340.append(p)
            else:
                others.append(p)
        return ch340, others

    @staticmethod
    def list_all_ports():
        return list(serial.tools.list_ports.comports())

    def connect(self, port=None):
        if port:
            self.port = port
        if not self.port:
            ch340, others = self.find_ch340_ports()
            if ch340:
                self.port = ch340[0].device
            elif others:
                self.port = others[0].device
            else:
                raise RuntimeError("No serial port found")

        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.05
        )
        return self.port

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None

    @staticmethod
    def parse_frame(frame):
        if len(frame) != 10:
            return None
        if frame[0] != 0x0A or frame[1] != 0x01:
            return None
        if frame[8] != 0x01 or frame[9] != 0x0A:
            return None

        adc_raw = (frame[2] << 16) | (frame[3] << 8) | frame[4]
        if adc_raw & 0x800000:
            adc_raw -= 0x1000000

        voltage = adc_raw * UartReader.VREF / (UartReader.GAIN * UartReader.ADC_MAX)
        return {
            'raw_bytes': bytes(frame),
            'middle': bytes(frame[2:8]),
            'adc_raw': adc_raw,
            'voltage': voltage,
            'timestamp': datetime.now()
        }

    def add_callback(self, callback):
        self._callbacks.append(callback)

    def remove_callback(self, callback):
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def start(self):
        if not self.ser or not self.ser.is_open:
            self.connect()
        self.running = True
        self._buffer = bytearray()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=2)
        self.disconnect()

    def __enter__(self):
        self.connect()
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def _read_loop(self):
        buf = self._buffer
        while self.running:
            try:
                waiting = self.ser.in_waiting
                if waiting:
                    chunk = self.ser.read(waiting)
                    buf.extend(chunk)
                else:
                    time.sleep(0.001)

                while len(buf) >= 10:
                    idx = buf.find(b'\x0a\x01')
                    if idx < 0:
                        if len(buf) > 1:
                            del buf[:len(buf) - 1]
                        break

                    if idx > 0:
                        del buf[:idx]

                    if len(buf) < 10:
                        break

                    if buf[8] == 0x01 and buf[9] == 0x0A:
                        frame = bytes(buf[:10])
                        result = self.parse_frame(frame)
                        if result:
                            with self.lock:
                                self.valid_count += 1
                                self.voltage = result['voltage']
                                self.last_raw = result['raw_bytes']
                                self.last_adc = result['adc_raw']
                            for cb in self._callbacks:
                                try:
                                    cb(result)
                                except Exception:
                                    pass
                        del buf[:10]
                    else:
                        with self.lock:
                            self.error_count += 1
                        next_idx = buf.find(b'\x0a\x01', 1)
                        if next_idx < 0:
                            if len(buf) > 1:
                                del buf[:len(buf) - 1]
                            break
                        del buf[:next_idx]

            except serial.SerialException:
                break
            except Exception:
                pass

    def get_voltage(self):
        with self.lock:
            return self.voltage

    def get_stats(self):
        with self.lock:
            return {
                'valid': self.valid_count,
                'error': self.error_count,
                'voltage': self.voltage,
                'raw': self.last_raw,
                'adc': self.last_adc,
            }

    def voltages(self):
        import queue
        q = queue.Queue(maxsize=1)

        def _cb(data):
            try:
                q.put_nowait(data['voltage'])
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(data['voltage'])
                except queue.Empty:
                    pass

        self.add_callback(_cb)
        self.start()
        while self.running:
            try:
                yield q.get(timeout=0.5)
            except queue.Empty:
                if not self.running:
                    break
        self.remove_callback(_cb)


def run_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox

    reader = UartReader()
    recording = False
    record_rows = []

    root = tk.Tk()
    root.title("UART Serial Assistant — CH340")
    root.geometry("600x550")
    root.minsize(500, 400)

    port_var = tk.StringVar()
    status_var = tk.StringVar(value="Disconnected")
    volt_var = tk.StringVar(value="0.000000000 V")
    stats_var = tk.StringVar(value="Valid: 0 | Error: 0")
    record_btn_text = tk.StringVar(value="Start Recording")

    def get_selected_device():
        sel = port_var.get()
        if sel:
            return sel.split(' ')[0]
        return None

    def refresh_ports():
        ch340, others = reader.find_ch340_ports()
        all_ports = reader.list_all_ports()
        values = []
        for p in ch340:
            values.append(f"{p.device} - {p.description or 'Unknown'} [CH340]")
        for p in others:
            values.append(f"{p.device} - {p.description or 'Unknown'}")
        combo['values'] = values
        if ch340:
            port_var.set(values[0])
        elif values:
            port_var.set(values[0])

    def connect_port():
        dev = get_selected_device()
        if not dev:
            messagebox.showwarning("Warning", "Please select a serial port")
            return
        try:
            reader.connect(dev)
            reader.start()
            status_var.set(f"Connected: {dev}")
            btn_connect.config(state=tk.DISABLED)
            btn_disconnect.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))

    def disconnect_port():
        reader.stop()
        status_var.set("Disconnected")
        btn_connect.config(state=tk.NORMAL)
        btn_disconnect.config(state=tk.DISABLED)
        volt_var.set("0.000000000 V")
        stats_var.set(f"Valid: {reader.valid_count} | Error: {reader.error_count}")

    def on_frame(data):
        volt_var.set(f"{data['voltage']:.9f} V")
        stats_var.set(f"Valid: {reader.valid_count} | Error: {reader.error_count}")

        ts = data['timestamp'].strftime("%H:%M:%S.%f")[:-3]
        line = f"[{ts}] {data['voltage']:+.9f} V\n"
        text_output.insert(tk.END, line)
        text_output.see(tk.END)

        if int(text_output.index('end-1c').split('.')[0]) > 800:
            text_output.delete('1.0', '2.0')

        nonlocal recording, record_rows
        if recording:
            record_rows.append((data['timestamp'], data['voltage'], data['adc_raw'], data['raw_bytes']))

    reader.add_callback(on_frame)

    def toggle_recording():
        nonlocal recording, record_rows
        if recording:
            recording = False
            record_btn_text.set("Start Recording")
            btn_save.config(state=tk.NORMAL)
        else:
            recording = True
            record_rows = []
            record_btn_text.set("Stop Recording")
            btn_save.config(state=tk.DISABLED)

    def save_to_desktop():
        nonlocal record_rows
        if not record_rows:
            messagebox.showwarning("Warning", "No recorded data to save")
            return
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        filename = f"uart_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = os.path.join(desktop, filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("Timestamp,Voltage(V),ADC_Raw,Hex_Frame\n")
                for ts, v, raw, frame in record_rows:
                    hex_str = ' '.join(f'{b:02X}' for b in frame)
                    f.write(f"{ts.isoformat()},{v:.9f},{raw},{hex_str}\n")
            messagebox.showinfo("Saved", f"Data saved to:\n{path}")
            record_rows = []
            btn_save.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    top = ttk.Frame(root, padding=5)
    top.pack(fill=tk.X)

    ttk.Label(top, text="Port:").pack(side=tk.LEFT)
    combo = ttk.Combobox(top, textvariable=port_var, width=25, state='readonly')
    combo.pack(side=tk.LEFT, padx=5)
    ttk.Button(top, text="Refresh", command=refresh_ports).pack(side=tk.LEFT, padx=2)
    btn_connect = ttk.Button(top, text="Connect", command=connect_port)
    btn_connect.pack(side=tk.LEFT, padx=2)
    btn_disconnect = ttk.Button(top, text="Disconnect", command=disconnect_port, state=tk.DISABLED)
    btn_disconnect.pack(side=tk.LEFT, padx=2)
    ttk.Label(top, textvariable=status_var, foreground='gray').pack(side=tk.LEFT, padx=10)

    volt_frame = ttk.LabelFrame(root, text="Voltage", padding=10)
    volt_frame.pack(fill=tk.X, padx=10, pady=(5, 0))
    ttk.Label(volt_frame, textvariable=volt_var, font=("Consolas", 28)).pack()
    ttk.Label(volt_frame, textvariable=stats_var, font=("Consolas", 9)).pack()

    out_frame = ttk.LabelFrame(root, text="Output", padding=3)
    out_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    text_output = tk.Text(out_frame, height=12, font=("Consolas", 10), state=tk.NORMAL, wrap=tk.NONE)
    text_output.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

    scroll_y = ttk.Scrollbar(out_frame, orient=tk.VERTICAL, command=text_output.yview)
    scroll_y.pack(fill=tk.Y, side=tk.RIGHT)
    text_output.config(yscrollcommand=scroll_y.set)

    btn_frame = ttk.Frame(root, padding=5)
    btn_frame.pack(fill=tk.X)
    btn_record = ttk.Button(btn_frame, textvariable=record_btn_text, command=toggle_recording)
    btn_record.pack(side=tk.LEFT, padx=2)
    btn_save = ttk.Button(btn_frame, text="Save to Desktop", command=save_to_desktop, state=tk.DISABLED)
    btn_save.pack(side=tk.LEFT, padx=2)

    refresh_ports()

    def on_close():
        reader.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('--help', '-h'):
        print("UART Serial Assistant")
        print("Usage:")
        print("  python uart_assistant.py              # GUI mode")
        print("  python uart_assistant.py --list       # List available ports")
        print("  python uart_assistant.py COM3         # Console mode, prints voltage")
        print()
        print("As Python module:")
        print("  from uart_assistant import UartReader")
        print("  reader = UartReader()")
        print("  reader.connect('COM3')")
        print("  reader.start()")
        print("  voltage = reader.get_voltage()  # poll latest voltage")
        print("  reader.stop()")
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        ch340, others = UartReader.find_ch340_ports()
        print("CH340 ports:")
        for p in ch340:
            print(f"  {p.device}  {p.description}")
        print("Other ports:")
        for p in others:
            print(f"  {p.device}  {p.description}")
        if not ch340 and not others:
            print("  (none)")
        sys.exit(0)

    if len(sys.argv) > 1:
        port = sys.argv[1]
        print(f"Connecting to {port}...")
        reader = UartReader()
        reader.connect(port)
        reader.start()
        try:
            while True:
                v = reader.get_voltage()
                s = reader.get_stats()
                print(f"\rValid:{s['valid']:6d}  Error:{s['error']:6d}  Voltage:{v:+.9f} V", end='')
                time.sleep(0.05)
        except KeyboardInterrupt:
            print()
        finally:
            reader.stop()
    else:
        run_gui()
