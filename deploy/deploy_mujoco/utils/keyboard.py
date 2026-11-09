"""Incremental velocity commands from the MuJoCo viewer key callback."""
from threading import Lock

import numpy as np


class KeyboardVelocity:
    """Press keys to change a persistent command; Space clears all three axes."""

    def __init__(self, maximum):
        self.maximum = np.asarray(maximum, dtype=np.float32).copy()
        self._command = np.zeros(3, dtype=np.float32)
        self._lock = Lock()

    def reset(self):
        with self._lock:
            self._command.fill(0.0)

    def read(self):
        with self._lock:
            return self._command.copy()

    def on_key(self, keycode):
        keys = {ord('W'): (0, 0.1), ord('S'): (0, -0.1),
                ord('A'): (1, 0.1), ord('D'): (1, -0.1),
                ord('Q'): (2, 0.1), ord('E'): (2, -0.1)}
        if ord('a') <= keycode <= ord('z'):
            keycode -= 32
        with self._lock:
            if keycode == ord(' '):
                self._command.fill(0.0)
            elif keycode in keys:
                axis, change = keys[keycode]
                self._command[axis] = np.clip(np.round(self._command[axis] + change, 4),
                                             -self.maximum[axis], self.maximum[axis])
            else:
                return
            command = self._command.copy()
        print(f'\n[KEYBOARD] vx={command[0]:+.2f} m/s  vy={command[1]:+.2f} m/s  yaw={command[2]:+.2f} rad/s', flush=True)
