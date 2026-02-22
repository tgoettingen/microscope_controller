from core.multiaxis import MultiAxisExperiment, MultiAxisRunner, XAxis, YAxis, ChannelAxis
from devices.base import Camera, StageXY, LightSource, FilterWheel
from core.experiment import ChannelConfig


def make_camera_scan(camera: Camera, stage: StageXY, light: LightSource, fw: FilterWheel):
    channels = [
        ChannelConfig(name="BF", filter_position=0, light_intensity=10.0, exposure_ms=20.0),
        ChannelConfig(name="GFP", filter_position=1, light_intensity=30.0, exposure_ms=50.0),
    ]

    axes = [
        ChannelAxis(camera, light, fw, channels, wait_s=0.01),  # inner
        XAxis(stage, start=0.0, end=1000.0, step=500.0),
        YAxis(stage, start=0.0, end=1000.0, step=500.0),
    ]

    def measure(state: dict):
        ch: ChannelConfig = state["Channel"]
        x = state["X"]
        y = state["Y"]
        img = camera.snap()
        meta = {
            "channel": ch.name,
            "x": x,
            "y": y,
            "round": state["round"],
        }
        # here you call your existing on_image / writer / GUI hooks
        print("Measured image", meta)

    exp = MultiAxisExperiment(axes=axes, measure=measure, n_rounds=1)
    return MultiAxisRunner(exp)