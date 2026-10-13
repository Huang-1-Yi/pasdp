"""Optional binary gripper on explicitly configured AUBO digital outputs.

Feedback is the output latch, NOT measured finger position. No pin, polarity,
tool voltage, or input/output direction is guessed or changed on startup.
SDK reference: https://docs.aubo-robotics.cn/arcs_api/en/group__IoControl.html
"""
from __future__ import annotations

import time

import numpy as np


class AuboDigitalGripper:
    def __init__(self, io, settings):
        self.io = io
        self.kind = settings.get("type", "disabled")
        self.mode = settings.get("mode", "binary")
        if self.mode not in ("binary", "independent"):
            raise ValueError("Gripper mode must be binary or independent")
        if self.kind not in ("disabled", "tool_digital", "standard_digital"):
            raise ValueError("Supported gripper types: disabled, tool_digital, standard_digital")
        self.pins = settings.get("pins", [settings.get("pin")])
        self.open_levels = settings.get("open_levels", [settings.get("open_level")])
        self.close_levels = settings.get("close_levels")
        self.last_command = None
        self.state = -1.
        self._next_feedback_poll = 0.
        if self.kind != "disabled":
            if (not isinstance(self.pins, list) or not 1 <= len(self.pins) <= 8 or
                    any(not isinstance(p, int) or isinstance(p, bool) or p < 0 for p in self.pins) or
                    len(set(self.pins)) != len(self.pins)):
                raise ValueError("Digital gripper requires 1..8 unique output pins")
            if self.mode == "binary":
                if (not isinstance(self.open_levels, list) or len(self.open_levels) != len(self.pins) or
                        any(not isinstance(v, bool) for v in self.open_levels)):
                    raise ValueError("Binary gripper requires boolean open_levels")
                if self.close_levels is None and len(self.pins) == 1:
                    self.close_levels = [not self.open_levels[0]]
                if (not isinstance(self.close_levels, list) or len(self.close_levels) != len(self.pins) or
                        any(not isinstance(v, bool) for v in self.close_levels) or
                        self.close_levels == self.open_levels):
                    raise ValueError("Multiple outputs require explicit, distinct close_levels and open_levels")
            self.prefix = "Tool" if self.kind == "tool_digital" else "Standard"

    @property
    def enabled(self):
        return self.kind != "disabled"

    def start(self):
        if not self.enabled:
            return
        if self.io is None:
            raise RuntimeError("AUBO IO interface unavailable")
        if max(self.pins) >= getattr(self.io, f"get{self.prefix}DigitalOutputNum")():
            raise ValueError("Configured gripper pin is out of range")
        if self.kind == "tool_digital" and any(self.io.isToolIoInput(p) for p in self.pins):
            raise RuntimeError("Gripper tool pin is configured as input; set it on the pendant")
        self.refresh()

    def refresh(self):
        if self.enabled:
            levels = [bool(getattr(self.io, f"get{self.prefix}DigitalOutput")(p)) for p in self.pins]
            self.state = (float(sum(int(v) << i for i, v in enumerate(levels))) if self.mode == "independent"
                          else (1. if levels == self.open_levels else (0. if levels == self.close_levels else -1.)))
        return self.state

    def poll_feedback(self):
        """Read an unconfirmed output latch at most every 100 ms; never write IO.

        A successful setter return is not latch feedback. The first read after a
        write can still show the previous level, so the servo loop must keep
        refreshing until the requested combination is observed. No sleeps,
        write retries, or target-derived feedback are introduced here.
        """
        if (self.enabled and self.last_command is not None and
                self.state != self.last_command and
                time.monotonic() >= self._next_feedback_poll):
            self.refresh()
            self._next_feedback_poll = time.monotonic() + .1
        return self.state

    def command(self, opened):
        if not self.enabled:
            raise RuntimeError("Gripper is disabled; no command can be sent")
        self.validate_command(opened)
        levels = ([bool(int(opened) & (1 << i)) for i in range(len(self.pins))] if self.mode == "independent"
                  else (self.open_levels if opened == 1. else self.close_levels))
        if opened != self.last_command:
            # Turn outputs off before turning others on. These writes are not atomic;
            # hardware interlocks are required for mutually exclusive valve inputs.
            for pin, level in sorted(zip(self.pins, levels), key=lambda item: item[1]):
                ret = getattr(self.io, f"set{self.prefix}DigitalOutput")(pin, level)
                if ret != 0:
                    raise RuntimeError(f"Gripper output {pin} rejected: {ret}")
            self.last_command = opened
        # An explicit repeated command also reads feedback; it does not resend
        # the same output writes. Never copy the requested value into state.
        self.refresh()
        self._next_feedback_poll = time.monotonic() + .1
        return self.state

    def validate_command(self, value):
        upper = (1 << len(self.pins)) - 1 if self.mode == "independent" else 1
        if not self.enabled or not np.isfinite(value) or value != int(value) or not 0 <= value <= upper:
            raise ValueError("Invalid digital output command")
