#!/usr/bin/env python3
"""
MSP432P401R Control Panel
=========================

A Tkinter + pyserial GUI for the ``MSP432-CTRL v1`` firmware in this repo.

It talks to the board over the LaunchPad backchannel UART (the XDS110
"Application/User UART" COM port) at 115200 baud, using the line-based ASCII
protocol implemented in ``src/main.c``.

Features
--------
* Connect / disconnect, with a serial-port picker and auto-refresh.
* One row per board channel (auto-discovered via ``LIST``):
    - ON / OFF / TOGGLE buttons
    - a live state indicator
    - a per-row "select" checkbox for group actions
* Pulse-train generator: on-time, off-time, and repeat count (0 = forever),
  applied to the selected channels (or a single channel).
* Software-PWM generator: frequency + duty %, for LED dimming / duty cycling.
* Group actions over the selected channels, plus a global ALL OFF panic button.
* A raw command entry + scrolling log of everything sent and received, so the
  protocol stays transparent and debuggable.

Run
---
    pip install -r requirements.txt
    python gui.py

No serial port? Launch with ``--simulate`` for a built-in fake board so you can
explore the UI without hardware.
"""

import argparse
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

try:
    import serial
    import serial.tools.list_ports
except ImportError:      # pragma: no cover - handled at runtime
    serial = None


BAUD = 115200
READ_TIMEOUT = 0.1


# ---------------------------------------------------------------------------
# Serial transport (real + simulated share one interface)
# ---------------------------------------------------------------------------

class SerialLink:
    """Threaded line-oriented serial link.

    Outgoing commands are written directly; incoming bytes are read on a
    background thread and pushed as whole lines onto ``rx_queue``.
    """

    def __init__(self, rx_queue):
        self.rx_queue = rx_queue
        self._ser = None
        self._reader = None
        self._stop = threading.Event()
        self._buf = bytearray()

    @property
    def is_open(self):
        return self._ser is not None

    def open(self, port):
        if serial is None:
            raise RuntimeError("pyserial not installed (pip install pyserial)")
        self._ser = serial.Serial(port, BAUD, timeout=READ_TIMEOUT)
        self._stop.clear()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def close(self):
        self._stop.set()
        if self._reader:
            self._reader.join(timeout=1.0)
            self._reader = None
        if self._ser:
            try:
                self._ser.close()
            finally:
                self._ser = None

    def send(self, line):
        if not self._ser:
            raise RuntimeError("not connected")
        data = (line.strip() + "\n").encode("ascii", "replace")
        self._ser.write(data)
        self.rx_queue.put(("tx", line.strip()))

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(256)
            except Exception as exc:                 # port yanked, etc.
                self.rx_queue.put(("err", f"read error: {exc}"))
                break
            if not chunk:
                continue
            self._buf.extend(chunk)
            while b"\n" in self._buf:
                raw, _, rest = self._buf.partition(b"\n")
                self._buf = bytearray(rest)
                text = raw.decode("ascii", "replace").strip("\r\n")
                if text:
                    self.rx_queue.put(("rx", text))


class SimLink(SerialLink):
    """In-process fake board that mimics the firmware protocol.

    Lets the GUI be exercised end-to-end with no hardware attached.
    """

    CHANNELS = ["LED1", "LED_R", "LED_G", "LED_B", "TTL0", "TTL1", "TTL2", "TTL3"]

    def __init__(self, rx_queue):
        super().__init__(rx_queue)
        self._open = False
        self._mode = ["static"] * len(self.CHANNELS)
        self._level = [0] * len(self.CHANNELS)

    @property
    def is_open(self):
        return self._open

    def open(self, port):
        self._open = True
        self.rx_queue.put(("rx", "MSP432-CTRL v1 READY [SIMULATED]"))

    def close(self):
        self._open = False

    def _emit(self, text):
        self.rx_queue.put(("rx", text))

    def _range(self, tok):
        if tok in ("ALL", "*"):
            return range(len(self.CHANNELS))
        if tok.isdigit() and int(tok) < len(self.CHANNELS):
            return [int(tok)]
        return None

    def send(self, line):
        line = line.strip()
        self.rx_queue.put(("tx", line))
        parts = line.split()
        if not parts:
            self._emit("ERR empty")
            return
        cmd = parts[0].upper()

        if cmd == "PING":
            self._emit("PONG"); self._emit("OK")
        elif cmd == "ID":
            self._emit("MSP432-CTRL v1"); self._emit("OK")
        elif cmd == "LIST":
            for i, n in enumerate(self.CHANNELS):
                self._emit(f"CH {i} {n}")
            self._emit("OK")
        elif cmd == "STATUS":
            for i, n in enumerate(self.CHANNELS):
                self._emit(f"ST {i} {self._mode[i]} {self._level[i]}")
            self._emit("OK")
        elif cmd == "ALLOFF":
            for i in range(len(self.CHANNELS)):
                self._mode[i] = "static"; self._level[i] = 0
            self._emit("OK")
        elif cmd in ("SET", "TOGGLE", "STOP", "PULSE", "PWM") and len(parts) >= 2:
            rng = self._range(parts[1].upper())
            if rng is None:
                self._emit("ERR bad channel"); return
            for i in rng:
                if cmd == "SET" and len(parts) >= 3:
                    self._mode[i] = "static"; self._level[i] = 1 if parts[2] == "1" else 0
                elif cmd == "TOGGLE":
                    self._mode[i] = "static"; self._level[i] ^= 1
                elif cmd == "STOP":
                    self._mode[i] = "static"; self._level[i] = 0
                elif cmd == "PULSE":
                    self._mode[i] = "pulse"; self._level[i] = 1
                elif cmd == "PWM":
                    self._mode[i] = "pwm"
            self._emit("OK")
        else:
            self._emit("ERR unknown cmd")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class ControlPanel(ttk.Frame):
    POLL_MS = 40

    def __init__(self, master, simulate=False):
        super().__init__(master, padding=8)
        self.simulate = simulate
        self.rx_queue = queue.Queue()
        self.link = SimLink(self.rx_queue) if simulate else SerialLink(self.rx_queue)

        self.channels = []          # list of dicts: name, index, widgets, vars
        self._chan_rows_built = False

        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._build_connection_bar()
        self._build_channel_area()
        self._build_generators()
        self._build_log()

        self.after(self.POLL_MS, self._pump_rx)
        if simulate:
            self._connect()      # auto-connect the fake board

    # -- connection bar ---------------------------------------------------

    def _build_connection_bar(self):
        bar = ttk.LabelFrame(self, text="Connection", padding=6)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        bar.columnconfigure(1, weight=1)

        ttk.Label(bar, text="Port:").grid(row=0, column=0, padx=(0, 4))
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(bar, textvariable=self.port_var, width=28,
                                       state="readonly")
        self.port_combo.grid(row=0, column=1, sticky="ew", padx=4)

        ttk.Button(bar, text="Refresh", command=self._refresh_ports).grid(row=0, column=2, padx=2)
        self.connect_btn = ttk.Button(bar, text="Connect", command=self._toggle_connect)
        self.connect_btn.grid(row=0, column=3, padx=2)

        self.status_var = tk.StringVar(value="Disconnected")
        ttk.Label(bar, textvariable=self.status_var, foreground="#a00").grid(
            row=0, column=4, padx=(8, 0))

        self._refresh_ports()

    def _refresh_ports(self):
        ports = []
        if serial is not None:
            ports = [p.device for p in serial.tools.list_ports.comports()]
        if self.simulate:
            ports = ["SIMULATED"] + ports
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])

    # -- channel table ----------------------------------------------------

    def _build_channel_area(self):
        wrap = ttk.LabelFrame(self, text="Channels", padding=6)
        wrap.grid(row=1, column=0, sticky="nsew", pady=6)
        wrap.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # header
        hdr = ttk.Frame(wrap)
        hdr.grid(row=0, column=0, sticky="ew")
        for col, (text, w) in enumerate(
                [("Sel", 4), ("#", 3), ("Name", 10), ("State", 8), ("Mode", 8), ("Actions", 20)]):
            ttk.Label(hdr, text=text, width=w, anchor="w",
                      font=("TkDefaultFont", 9, "bold")).grid(row=0, column=col, padx=2)

        self.rows_frame = ttk.Frame(wrap)
        self.rows_frame.grid(row=1, column=0, sticky="nsew")

        # group actions
        grp = ttk.Frame(wrap)
        grp.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(grp, text="Selected:").pack(side="left", padx=(0, 4))
        ttk.Button(grp, text="All",  width=5, command=lambda: self._select_all(True)).pack(side="left", padx=1)
        ttk.Button(grp, text="None", width=5, command=lambda: self._select_all(False)).pack(side="left", padx=1)
        ttk.Separator(grp, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(grp, text="ON",     command=lambda: self._group("SET", "1")).pack(side="left", padx=1)
        ttk.Button(grp, text="OFF",    command=lambda: self._group("SET", "0")).pack(side="left", padx=1)
        ttk.Button(grp, text="Toggle", command=lambda: self._group("TOGGLE")).pack(side="left", padx=1)
        ttk.Button(grp, text="Stop",   command=lambda: self._group("STOP")).pack(side="left", padx=1)

        self.panic_btn = ttk.Button(wrap, text="⏻  ALL OFF (panic)",
                                    command=lambda: self._send("ALLOFF"))
        self.panic_btn.grid(row=3, column=0, sticky="ew", pady=(6, 0))

    def _build_channel_rows(self):
        for w in self.rows_frame.winfo_children():
            w.destroy()
        for ch in self.channels:
            r = ch["index"]
            row = ttk.Frame(self.rows_frame)
            row.grid(row=r, column=0, sticky="ew", pady=1)

            sel = tk.BooleanVar(value=False)
            ch["sel"] = sel
            ttk.Checkbutton(row, variable=sel, width=3).grid(row=0, column=0, padx=2)
            ttk.Label(row, text=str(r), width=3, anchor="w").grid(row=0, column=1, padx=2)
            ttk.Label(row, text=ch["name"], width=10, anchor="w").grid(row=0, column=2, padx=2)

            state = tk.Canvas(row, width=16, height=16, highlightthickness=0)
            dot = state.create_oval(2, 2, 14, 14, fill="#444", outline="#222")
            state.grid(row=0, column=3, padx=(6, 2))
            ch["canvas"], ch["dot"] = state, dot

            ch["mode_var"] = tk.StringVar(value="static")
            ttk.Label(row, textvariable=ch["mode_var"], width=8, anchor="w").grid(row=0, column=4, padx=2)

            ttk.Button(row, text="On",     width=4,
                       command=lambda i=r: self._send(f"SET {i} 1")).grid(row=0, column=5, padx=1)
            ttk.Button(row, text="Off",    width=4,
                       command=lambda i=r: self._send(f"SET {i} 0")).grid(row=0, column=6, padx=1)
            ttk.Button(row, text="Toggle", width=6,
                       command=lambda i=r: self._send(f"TOGGLE {i}")).grid(row=0, column=7, padx=1)
        self._chan_rows_built = True

    # -- pulse / pwm generators ------------------------------------------

    def _build_generators(self):
        gens = ttk.Frame(self)
        gens.grid(row=2, column=0, sticky="ew", pady=6)
        gens.columnconfigure(0, weight=1)
        gens.columnconfigure(1, weight=1)

        # Pulse train
        pf = ttk.LabelFrame(gens, text="Pulse train", padding=6)
        pf.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        self.pulse_on = tk.StringVar(value="200")
        self.pulse_off = tk.StringVar(value="200")
        self.pulse_cnt = tk.StringVar(value="0")
        self._labeled_entry(pf, "On (ms):",  self.pulse_on,  0)
        self._labeled_entry(pf, "Off (ms):", self.pulse_off, 1)
        self._labeled_entry(pf, "Repeat (0=∞):", self.pulse_cnt, 2)
        ttk.Button(pf, text="Apply to selected",
                   command=self._apply_pulse).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        # PWM
        wf = ttk.LabelFrame(gens, text="PWM (dimming / duty cycle)", padding=6)
        wf.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        self.pwm_freq = tk.StringVar(value="1000")
        self.pwm_duty = tk.StringVar(value="50")
        self._labeled_entry(wf, "Freq (Hz):", self.pwm_freq, 0)
        row = ttk.Frame(wf)
        row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Label(row, text="Duty %:", width=12, anchor="w").pack(side="left")
        self.duty_scale = ttk.Scale(row, from_=0, to=100, orient="horizontal",
                                    command=self._on_duty_slide)
        self.duty_scale.set(50)
        self.duty_scale.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Label(row, textvariable=self.pwm_duty, width=4).pack(side="left")
        ttk.Button(wf, text="Apply to selected",
                   command=self._apply_pwm).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))

    def _labeled_entry(self, parent, label, var, row):
        ttk.Label(parent, text=label, width=12, anchor="w").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var, width=10).grid(row=row, column=1, sticky="w", pady=2)

    def _on_duty_slide(self, val):
        self.pwm_duty.set(str(int(float(val))))

    # -- log / raw command -----------------------------------------------

    def _build_log(self):
        lf = ttk.LabelFrame(self, text="Log", padding=6)
        lf.grid(row=3, column=0, sticky="nsew", pady=(6, 0))
        lf.columnconfigure(0, weight=1)
        lf.rowconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        self.log = tk.Text(lf, height=9, wrap="none", state="disabled",
                           font=("Consolas", 9), background="#111", foreground="#ddd")
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log["yscrollcommand"] = sb.set
        self.log.tag_config("tx", foreground="#6cf")
        self.log.tag_config("rx", foreground="#9e9")
        self.log.tag_config("err", foreground="#f88")

        cmd = ttk.Frame(lf)
        cmd.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        cmd.columnconfigure(0, weight=1)
        self.cmd_var = tk.StringVar()
        entry = ttk.Entry(cmd, textvariable=self.cmd_var)
        entry.grid(row=0, column=0, sticky="ew")
        entry.bind("<Return>", lambda _e: self._send_raw())
        ttk.Button(cmd, text="Send", command=self._send_raw).grid(row=0, column=1, padx=(4, 0))

    # -- connection logic -------------------------------------------------

    def _toggle_connect(self):
        if self.link.is_open:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self.port_var.get()
        if not port:
            messagebox.showwarning("No port", "Select a serial port first.")
            return
        try:
            self.link.open(port)
        except Exception as exc:
            messagebox.showerror("Connect failed", str(exc))
            return
        self.status_var.set(f"Connected: {port}")
        self.connect_btn["text"] = "Disconnect"
        # discover channels + current state
        self.channels = []
        self._pending_list = []
        self.after(150, lambda: self._send("LIST"))
        self.after(350, lambda: self._send("STATUS"))

    def _disconnect(self):
        self.link.close()
        self.status_var.set("Disconnected")
        self.connect_btn["text"] = "Connect"

    # -- sending ----------------------------------------------------------

    def _send(self, line):
        if not self.link.is_open:
            self._log("err", "! not connected")
            return
        try:
            self.link.send(line)
        except Exception as exc:
            self._log("err", f"! send failed: {exc}")

    def _send_raw(self):
        text = self.cmd_var.get().strip()
        if text:
            self._send(text)
            self.cmd_var.set("")

    def _selected_indices(self):
        return [c["index"] for c in self.channels if c.get("sel") and c["sel"].get()]

    def _group(self, cmd, *args):
        idxs = self._selected_indices()
        if not idxs:
            self._log("err", "! no channels selected")
            return
        suffix = (" " + " ".join(args)) if args else ""
        for i in idxs:
            self._send(f"{cmd} {i}{suffix}")

    def _select_all(self, value):
        for c in self.channels:
            if c.get("sel"):
                c["sel"].set(value)

    def _apply_pulse(self):
        idxs = self._selected_indices()
        if not idxs:
            self._log("err", "! select channels for the pulse train")
            return
        try:
            on = int(self.pulse_on.get()); off = int(self.pulse_off.get())
            cnt = int(self.pulse_cnt.get())
            assert on >= 1 and off >= 0 and cnt >= 0
        except (ValueError, AssertionError):
            messagebox.showerror("Bad pulse values",
                                 "On >= 1 ms, Off >= 0 ms, Repeat >= 0 (0 = forever).")
            return
        for i in idxs:
            self._send(f"PULSE {i} {on} {off} {cnt}")

    def _apply_pwm(self):
        idxs = self._selected_indices()
        if not idxs:
            self._log("err", "! select channels for PWM")
            return
        try:
            freq = int(self.pwm_freq.get()); duty = int(self.pwm_duty.get())
            assert 1 <= freq <= 5000 and 0 <= duty <= 100
        except (ValueError, AssertionError):
            messagebox.showerror("Bad PWM values", "Freq 1..5000 Hz, Duty 0..100 %.")
            return
        for i in idxs:
            self._send(f"PWM {i} {freq} {duty}")

    # -- incoming pump ----------------------------------------------------

    def _pump_rx(self):
        try:
            while True:
                kind, text = self.rx_queue.get_nowait()
                if kind == "tx":
                    self._log("tx", f">> {text}")
                elif kind == "err":
                    self._log("err", f"! {text}")
                    self._disconnect()
                else:
                    self._log("rx", f"<< {text}")
                    self._handle_rx(text)
        except queue.Empty:
            pass
        self.after(self.POLL_MS, self._pump_rx)

    def _handle_rx(self, text):
        parts = text.split()
        if not parts:
            return
        tag = parts[0]
        if tag == "CH" and len(parts) >= 3 and parts[1].isdigit():
            idx = int(parts[1])
            if not any(c["index"] == idx for c in self.channels):
                self.channels.append({"index": idx, "name": parts[2]})
                self.channels.sort(key=lambda c: c["index"])
                self._build_channel_rows()
        elif tag == "ST" and len(parts) >= 4 and parts[1].isdigit():
            idx = int(parts[1])
            for c in self.channels:
                if c["index"] == idx and "canvas" in c:
                    c["mode_var"].set(parts[2])
                    self._set_dot(c, parts[3] == "1", parts[2])

    def _set_dot(self, ch, on, mode):
        color = {"static": "#3f6", "pulse": "#fd3", "pwm": "#3cf"}.get(mode, "#3f6")
        ch["canvas"].itemconfig(ch["dot"], fill=color if on else "#444")

    # -- logging ----------------------------------------------------------

    def _log(self, tag, text):
        self.log["state"] = "normal"
        stamp = time.strftime("%H:%M:%S")
        self.log.insert("end", f"{stamp} {text}\n", tag)
        self.log.see("end")
        self.log["state"] = "disabled"


def main():
    ap = argparse.ArgumentParser(description="MSP432P401R serial control panel")
    ap.add_argument("--simulate", action="store_true",
                    help="run against a built-in fake board (no hardware needed)")
    args = ap.parse_args()

    if serial is None and not args.simulate:
        print("pyserial is not installed. Run:  pip install -r requirements.txt\n"
              "Or explore the UI with:  python gui.py --simulate", file=sys.stderr)

    root = tk.Tk()
    root.title("MSP432P401R Control Panel")
    root.minsize(640, 720)
    panel = ControlPanel(root, simulate=args.simulate)
    root.protocol("WM_DELETE_WINDOW", lambda: (panel.link.close(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
