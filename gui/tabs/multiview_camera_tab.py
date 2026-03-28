from __future__ import annotations

import time
from typing import Any

from PyQt6 import QtWidgets, QtCore
import numpy as np
import pyqtgraph as pg


class _ImageTile(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.title = QtWidgets.QLabel("-")
        self.title.setWordWrap(True)
        layout.addWidget(self.title)

        self.glw = pg.GraphicsLayoutWidget()
        self.view = self.glw.addViewBox(row=0, col=0)
        self.view.setAspectLocked(True)
        self.view.invertY(False)
        self.img_item = pg.ImageItem()
        self.view.addItem(self.img_item)
        layout.addWidget(self.glw, 1)

    def set_image(self, img: Any, label: str) -> None:
        self.title.setText(label)
        if img is None:
            return

        try:
            arr = np.asarray(img)
        except Exception:
            return

        try:
            # pyqtgraph expects either 2D (H,W) or 3D (H,W,3)
            self.img_item.setImage(arr, autoLevels=True)
        except Exception:
            return


class MultiViewCameraTab(QtWidgets.QWidget):
    """Multi-view camera panel.

    Shows the last N captured frames in a small grid, intended for use during
    scans where motor moves and camera captures should be kept in lockstep.
    """

    enabled_changed = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None, n_views: int = 4):
        super().__init__(parent)
        self._n_views = max(1, int(n_views))
        self._next_idx = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QHBoxLayout()
        self.enable_check = QtWidgets.QCheckBox("Sync capture")
        self.enable_check.setChecked(True)
        self.clear_btn = QtWidgets.QPushButton("Clear")
        controls.addWidget(self.enable_check)
        controls.addStretch(1)
        controls.addWidget(self.clear_btn)
        layout.addLayout(controls)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        layout.addLayout(grid, 1)

        self._tiles: list[_ImageTile] = []
        n = self._n_views
        cols = 2 if n > 1 else 1
        rows = (n + cols - 1) // cols

        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= n:
                    break
                tile = _ImageTile()
                self._tiles.append(tile)
                grid.addWidget(tile, r, c)
                idx += 1

        self.enable_check.toggled.connect(lambda v: self.enabled_changed.emit(bool(v)))
        self.clear_btn.clicked.connect(self.clear)

    def is_enabled(self) -> bool:
        try:
            return bool(self.enable_check.isChecked())
        except Exception:
            return True

    def clear(self) -> None:
        self._next_idx = 0
        for t in self._tiles:
            try:
                t.set_image(None, "-")
            except Exception:
                pass

    def _format_label(self, meta: dict) -> str:
        try:
            ts = float(meta.get("timestamp", time.time()))
        except Exception:
            ts = time.time()
        try:
            state = meta.get("state") if isinstance(meta, dict) else None
        except Exception:
            state = None

        parts = [time.strftime("%H:%M:%S", time.localtime(ts))]
        if isinstance(state, dict):
            for k in ("X", "Y", "Z"):
                if k in state and state[k] is not None:
                    try:
                        parts.append(f"{k}={float(state[k]):.3f}")
                    except Exception:
                        parts.append(f"{k}={state[k]}")
            if "Channel" in state and state["Channel"] is not None:
                try:
                    ch = state["Channel"]
                    name = getattr(ch, "name", str(ch))
                    parts.append(str(name))
                except Exception:
                    pass
        return "  ".join(parts)

    @QtCore.pyqtSlot(object, dict)
    def add_image(self, img: object, meta: dict) -> None:
        if not self._tiles:
            return

        label = self._format_label(meta if isinstance(meta, dict) else {})
        idx = self._next_idx % len(self._tiles)
        self._next_idx += 1
        try:
            self._tiles[idx].set_image(img, label)
        except Exception:
            pass
