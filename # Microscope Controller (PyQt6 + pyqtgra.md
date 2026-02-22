# Microscope Controller (PyQt6 + pyqtgraph)

This is a minimal but structured microscope control framework with:

- Layered architecture (devices, core, GUI)
- Simulation mode (mock hardware)
- Experiment setup (time-lapse + z-stack)
- Live camera image display
- Live detector time-series plot
- PyQt6 + pyqtgraph GUI with tabs and menus

It is intended as a starting point for building a real microscope automation system.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt