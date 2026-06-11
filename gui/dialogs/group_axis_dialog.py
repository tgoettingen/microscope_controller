from PyQt6 import QtWidgets


class GroupAxisDialog(QtWidgets.QDialog):
    """Configure how an axis participates in the multi-axis scan.

    Two independent controls:

    - **Collapse to one step** (checkbox): when checked, the axis is not a scan
      dimension — all of its movements run inside a single step of the scan.
    - **Group with** (dropdown): choose ``(no group)`` to keep the axis as an
      independent dimension, or pick another axis to group with it. When grouped
      the user also chooses:
        - scan mode: ``sync`` (members advance together, in lockstep) or
          ``sequential`` (members run one after another).
        - overall steps: ``shorter`` (truncate to the member with fewer steps)
          or ``longer`` (extend to the member with more steps; the shorter one
          holds its last position).

    The two controls are independent and may be combined: an axis can be both
    collapsed into a single step *and* grouped with another axis.
    """

    def __init__(self, this_label: str, other_axes: list[tuple[int, str]],
                 parent=None, mode: str = "sync", length: str = "longer",
                 current: tuple | None = None, collapsed: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Group Axis")

        layout = QtWidgets.QFormLayout(self)

        info = QtWidgets.QLabel(f"Configure grouping for <b>{this_label}</b>.")
        info.setWordWrap(True)
        layout.addRow(info)

        # ── Collapse to one step (standalone option) ─────────────────────────
        self.onestep_cb = QtWidgets.QCheckBox("Collapse this axis into a single scan step")
        layout.addRow(self.onestep_cb)

        # ── "Group with" target ──────────────────────────────────────────────
        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.addItem("(no group)", ("none",))
        for idx, label in other_axes:
            self.target_combo.addItem(label, ("axis", idx))

        # Preselect the current choice. Collapse and grouping are independent.
        if collapsed:
            self.onestep_cb.setChecked(True)
        if current and current[0] == "axis":
            for i in range(self.target_combo.count()):
                data = self.target_combo.itemData(i)
                if data and data[0] == "axis" and data[1] == current[1]:
                    self.target_combo.setCurrentIndex(i)
                    break

        layout.addRow("Group with", self.target_combo)

        # ── Scan mode / overall steps (only for axis grouping) ───────────────
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["sync", "sequential"])
        idx = self.mode_combo.findText(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

        self.length_combo = QtWidgets.QComboBox()
        self.length_combo.addItems(["shorter", "longer"])
        idx = self.length_combo.findText(length)
        if idx >= 0:
            self.length_combo.setCurrentIndex(idx)

        layout.addRow("Scan mode", self.mode_combo)
        layout.addRow("Overall steps", self.length_combo)

        def _update_enabled():
            collapsed = self.onestep_cb.isChecked()
            # Collapse and grouping are independent and may be combined, so the
            # "Group with" dropdown stays available regardless of collapse.
            choice = self.target_combo.currentData()
            is_axis = bool(choice) and choice[0] == "axis"
            # Scan mode / overall steps apply to either a collapsed sweep or a
            # grouped axis.
            mode_active = collapsed or is_axis
            self.mode_combo.setEnabled(mode_active)
            # The shorter/longer choice only matters for synchronized scans.
            self.length_combo.setEnabled(mode_active and self.mode_combo.currentText() == "sync")

        self.onestep_cb.toggled.connect(lambda _c: _update_enabled())
        self.target_combo.currentIndexChanged.connect(lambda _i: _update_enabled())
        self.mode_combo.currentTextChanged.connect(lambda _t: _update_enabled())
        _update_enabled()

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get_choice(self) -> tuple:
        """Return the selected grouping target.

        One of: ``("none",)`` or ``("axis", index)``. This is independent of the
        collapse checkbox — use :meth:`is_collapsed` for that.
        """
        data = self.target_combo.currentData()
        return data if data else ("none",)

    def is_collapsed(self) -> bool:
        """Whether the axis should be collapsed into a single scan step."""
        return self.onestep_cb.isChecked()

    def get_values(self) -> tuple[str, str]:
        return self.mode_combo.currentText(), self.length_combo.currentText()

