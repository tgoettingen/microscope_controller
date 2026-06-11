"""CLI tool to verify multi-axis X-stage movement (incl. collapse / grouping).

This runs the *real* scan engine (``core.multiaxis``) against a simulated XY
stage so you can confirm the physical move order without any hardware or GUI.

It is especially useful to check "back-and-forth" setups where one X axis sweeps
forward, a second X axis sweeps backward, both are collapsed into a single scan
step and grouped sequentially.

Usage examples
--------------
Back-and-forth 0->100->0 with step 10 (two collapsed+grouped X axes):

    python -m scripts.check_axis_moves --back-and-forth 0 100 10

Back-and-forth nested inside an outer Z sweep 0->5 step 1 (repeats per Z):

    python -m scripts.check_axis_moves --back-and-forth 0 100 10 --with-z 0 5 1

Simple forward sweep only:

    python -m scripts.check_axis_moves --forward 0 100 10

Load an experiment JSON (uses its ``multiaxis.axes`` block, X + Z axes):

    python -m scripts.check_axis_moves --experiment experiments/group_backandforth.json

The tool prints every physical device move in order, the scan steps (measure
points), and the per-step state reported to the runner. With ``--expect`` you can
assert the exact ordered list of X targets and get a non-zero exit code on
mismatch (handy for CI / regression checks).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make the project root importable when run as a plain script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.multiaxis import (  # noqa: E402
    XAxis,
    ZAxis,
    GroupedAxis,
    OneStepAxis,
    MultiAxisExperiment,
    MultiAxisRunner,
)


class RecordingStage:
    """Minimal XY stage that records every move_to into a shared labelled log."""

    def __init__(self, log: list[tuple[str, float]]):
        self.x = 0.0
        self.y = 0.0
        self._log = log

    def connect(self):
        pass

    def move_to(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)
        self._log.append(("X", self.x))

    def get_position(self):
        return self.x, self.y

    def stop(self):
        pass


class RecordingFocus:
    """Minimal focus (Z) device that records every move_to into the shared log."""

    def __init__(self, log: list[tuple[str, float]]):
        self.z = 0.0
        self._log = log

    def connect(self):
        pass

    def move_to(self, z: float):
        self.z = float(z)
        self._log.append(("Z", self.z))

    def get_position(self):
        return self.z

    def stop(self):
        pass


def _build_axis(stage, focus, params: dict):
    """Build an XAxis or ZAxis from a param dict (params carry ``axis_type``)."""
    t = params.get("axis_type", "X")
    if t == "Z":
        return ZAxis(
            focus,
            params["start"],
            params["end"],
            params["step"],
            motor_devices=[focus],
            motor_mode=params.get("motor_mode", "sequential"),
            wait_s=params.get("wait", 0.0),
            sync_timeout=params.get("sync_timeout", 5.0),
            sync_poll=params.get("sync_poll", 0.01),
            sync_tol=params.get("sync_tol", 1e-3),
        )
    return XAxis(
        stage,
        params["start"],
        params["end"],
        params["step"],
        motor_devices=[stage],
        motor_mode=params.get("motor_mode", "sequential"),
        wait_s=params.get("wait", 0.0),
        sync_timeout=params.get("sync_timeout", 5.0),
        sync_poll=params.get("sync_poll", 0.01),
        sync_tol=params.get("sync_tol", 1e-3),
    )


def _apply_grouping(axes: list, cfgs: list[dict]) -> list:
    """Replicate MainWindow._apply_axis_grouping.

    - ``collapse_one_step`` wraps the axis in a OneStepAxis.
    - ``group_with_prev`` joins the axis into the previous group; multi-member
      groups become a GroupedAxis with the group's mode/length.
    """
    prepped = []
    for ax, params in zip(axes, cfgs):
        if params.get("collapse_one_step"):
            prepped.append(OneStepAxis(ax))
        else:
            prepped.append(ax)

    groups: list[list] = []
    group_meta: list[tuple[str, str]] = []
    for ax, params in zip(prepped, cfgs):
        join = bool(params.get("group_with_prev")) and bool(groups)
        meta = (
            params.get("group_mode", "sync"),
            params.get("group_length", "longer"),
        )
        if join:
            groups[-1].append(ax)
            group_meta[-1] = meta
        else:
            groups.append([ax])
            group_meta.append(meta)

    result = []
    for members, (mode, length) in zip(groups, group_meta):
        if len(members) == 1:
            result.append(members[0])
        else:
            result.append(GroupedAxis(members, mode=mode, length=length))
    return result


def _cfgs_from_args(args) -> list[dict]:
    """Build the ordered list of axis param dicts from CLI arguments.

    Each dict carries an ``axis_type`` key ("X" or "Z"). When ``--with-z`` is
    given it is prepended as the outer (first) scan dimension so the inner X
    back-and-forth repeats once per Z position.
    """
    if args.experiment:
        with open(args.experiment, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        axes = (data.get("multiaxis") or {}).get("axes") or []
        cfgs = []
        for a in axes:
            t = a.get("axis_type")
            if t not in ("X", "Z"):
                print(f"  (skipping unsupported axis: {t})")
                continue
            p = dict(a.get("params") or {})
            p["axis_type"] = t
            cfgs.append(p)
        if not cfgs:
            raise SystemExit("No X/Z axes found in the experiment's multiaxis.axes block.")
        return cfgs

    cfgs: list[dict] = []

    # Optional outer Z axis (prepended → outermost loop).
    if args.with_z:
        zs, ze, zstep = args.with_z
        cfgs.append({"axis_type": "Z", "start": zs, "end": ze, "step": zstep})

    if args.back_and_forth:
        start, end, step = args.back_and_forth
        cfgs.append({
            "axis_type": "X",
            "start": start,
            "end": end,
            "step": step,
            "collapse_one_step": True,
            "group_mode": "sequential",
            "group_length": "longer",
        })
        cfgs.append({
            "axis_type": "X",
            "start": end,
            "end": start,
            "step": step,
            "collapse_one_step": True,
            "group_with_prev": True,
            "group_mode": "sequential",
            "group_length": "longer",
        })
        return cfgs

    if args.forward:
        start, end, step = args.forward
        cfgs.append({"axis_type": "X", "start": start, "end": end, "step": step})
        return cfgs

    raise SystemExit("Specify one of --back-and-forth, --forward, or --experiment.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_axis_moves",
        description="Verify multi-axis stage movement (collapse / grouping / nesting).",
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--back-and-forth", nargs=3, type=float, metavar=("START", "END", "STEP"),
        help="Forward START->END then backward END->START, collapsed+grouped sequentially.",
    )
    g.add_argument(
        "--forward", nargs=3, type=float, metavar=("START", "END", "STEP"),
        help="Single forward sweep START->END.",
    )
    g.add_argument(
        "--experiment", type=str, metavar="JSON",
        help="Path to an experiment JSON; uses its multiaxis.axes X/Z axes verbatim.",
    )
    parser.add_argument(
        "--with-z", nargs=3, type=float, metavar=("START", "END", "STEP"),
        help="Add an outer Z axis (the X sweep repeats once per Z position).",
    )
    parser.add_argument(
        "--expect", type=str, default=None,
        help="Comma-separated expected ordered X targets; exits non-zero on mismatch.",
    )
    args = parser.parse_args(argv)

    cfgs = _cfgs_from_args(args)

    print("Axis configuration:")
    for i, c in enumerate(cfgs):
        flags = []
        if c.get("collapse_one_step"):
            flags.append("collapse")
        if c.get("group_with_prev"):
            flags.append(f"group({c.get('group_mode', 'sync')}/{c.get('group_length', 'longer')})")
        flag_s = (" [" + ", ".join(flags) + "]") if flags else ""
        print(f"  axis {i}: {c['axis_type']} {c['start']} -> {c['end']} "
              f"step {c['step']}{flag_s}")
    print()

    moves: list[tuple[str, float]] = []
    stage = RecordingStage(moves)
    focus = RecordingFocus(moves)
    raw_axes = [_build_axis(stage, focus, c) for c in cfgs]
    axes = _apply_grouping(raw_axes, cfgs)

    print(f"Built {len(axes)} scan dimension(s) after grouping:")
    for ax in axes:
        print(f"  - {type(ax).__name__}: {ax.name()}")
    print()

    steps: list[dict] = []

    def measure(state: dict):
        steps.append({"state": dict(state), "x": stage.x, "z": focus.z})

    runner = MultiAxisRunner(MultiAxisExperiment(axes=axes, measure=measure))
    runner.run()

    print("Physical moves (axis=target, in order):")
    print("  " + " -> ".join(f"{lbl}={val:g}" for lbl, val in moves))
    print()

    x_moves = [val for lbl, val in moves if lbl == "X"]

    print(f"Scan steps (measure points): {len(steps)}")
    for i, s in enumerate(steps):
        x = s["state"].get("X", s["state"].get("x"))
        z = s["state"].get("Z", s["state"].get("z"))
        try:
            x_s = f"{float(x):g}"
        except (TypeError, ValueError):
            x_s = repr(x)
        z_s = f"{float(z):g}" if z is not None else "-"
        print(f"  step {i}: reported Z={z_s} X={x_s}  "
              f"(devices at Z={s['z']:g} X={s['x']:g})")
    print()

    # Count how many back-and-forth round trips happened (one per Z position).
    if args.back_and_forth:
        n_z = 1
        if args.with_z:
            zs, ze, zstep = args.with_z
            mag = abs(zstep) or 1
            n_z = int(round(abs(ze - zs) / mag)) + 1
        print(f"Back-and-forth round trips expected: {n_z} (one per Z position)")
        print()

    ok = True
    if args.expect is not None:
        expected = [float(v) for v in args.expect.split(",") if v.strip() != ""]
        if x_moves == expected:
            print("EXPECT: PASS — X moves match expected sequence.")
        else:
            ok = False
            print("EXPECT: FAIL")
            print(f"  expected: {expected}")
            print(f"  actual  : {x_moves}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
