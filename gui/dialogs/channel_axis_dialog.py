from PyQt6 import QtWidgets

from core.experiment import ChannelConfig
from core.multiaxis import AxisConfig


class ChannelAxisDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, config: AxisConfig | None = None):
        super().__init__(parent)
        self.setWindowTitle("Channel Axis Settings")

        layout = QtWidgets.QVBoxLayout(self)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "Filter", "Intensity", "Exposure"])
        layout.addWidget(self.table)

        add_btn = QtWidgets.QPushButton("Add Channel")
        layout.addWidget(add_btn)

        self.wait_spin = QtWidgets.QDoubleSpinBox()
        self.wait_spin.setRange(0.0, 10.0)
        self.wait_spin.setValue(0.0)
        layout.addWidget(QtWidgets.QLabel("Wait [s]"))
        layout.addWidget(self.wait_spin)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(btns)

        add_btn.clicked.connect(self._add_channel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        if config is not None:
            self.wait_spin.setValue(float(config.params.get("wait", self.wait_spin.value())))
            for channel in config.params.get("channels", []):
                if isinstance(channel, dict):
                    values = (
                        channel.get("name", ""),
                        channel.get("filter_position", channel.get("filter", "")),
                        channel.get("light_intensity", channel.get("intensity", "")),
                        channel.get("exposure_ms", channel.get("exposure", "")),
                    )
                else:
                    values = (
                        getattr(channel, "name", ""),
                        getattr(channel, "filter_position", ""),
                        getattr(channel, "light_intensity", ""),
                        getattr(channel, "exposure_ms", ""),
                    )
                row = self.table.rowCount()
                self.table.insertRow(row)
                for column, value in enumerate(values):
                    self.table.setItem(
                        row,
                        column,
                        QtWidgets.QTableWidgetItem(str(value)),
                    )

    def _add_channel(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col in range(4):
            self.table.setItem(row, col, QtWidgets.QTableWidgetItem(""))

    def get_config(self) -> AxisConfig:
        channels: list[ChannelConfig] = []
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            filt_item = self.table.item(row, 1)
            inten_item = self.table.item(row, 2)
            exp_item = self.table.item(row, 3)
            if not all((name_item, filt_item, inten_item, exp_item)):
                continue
            name = name_item.text()
            filt = int(filt_item.text())
            inten = float(inten_item.text())
            exp = float(exp_item.text())
            channels.append(ChannelConfig(name, filt, inten, exp))

        return AxisConfig(
            axis_type="Channel",
            params={
                "channels": channels,
                "wait": self.wait_spin.value(),
            },
        )