"""Real-time recognition loop and event de-bouncing.

Replaces the original ``real_time_recognize`` method. Two ideas carried over
from the original because they were genuinely good:

* **Temporal de-bouncing** — only emit an identity event once the same person
  has been seen for several consecutive frames, which kills single-frame
  flicker and false positives.
* **Kiosk websocket hook** — the same ``best_match`` JSON message the original
  Jetson Nano sent to the Django kiosk server, so this drops into the existing
  ``aisecurity_server`` deployment unchanged.
"""

from __future__ import annotations

import json
import time
from collections import deque
from typing import Callable, Deque, Optional

import numpy as np

from aisecurity.recognizer import Identification, Recognizer


class EventDebouncer:
    """Fires a callback once a name is stable over a sliding window."""

    def __init__(self, window: int = 8, min_hits: int = 5, cooldown_s: float = 5.0):
        self.window = window
        self.min_hits = min_hits
        self.cooldown_s = cooldown_s
        self._recent: Deque[Optional[str]] = deque(maxlen=window)
        self._last_fired: dict[str, float] = {}

    def update(self, name: Optional[str]) -> Optional[str]:
        """Feed the current frame's top name; returns a name when it commits."""
        self._recent.append(name)
        if name is None:
            return None
        if self._recent.count(name) < self.min_hits:
            return None
        now = time.monotonic()
        if now - self._last_fired.get(name, 0.0) < self.cooldown_s:
            return None
        self._last_fired[name] = now
        return name


def stream(
    recognizer: Recognizer,
    *,
    source: int | str = 0,
    show: bool = True,
    on_event: Optional[Callable[[str], None]] = None,
    socket=None,
    require_frontal: bool = True,
) -> None:
    """Run live recognition from a webcam/video ``source``.

    :param on_event: called with a name once it is committed by the debouncer.
    :param socket: optional websocket-like object with ``.send(str)``; receives
        ``{"best_match": name}`` — wire-compatible with the kiosk server.
    """
    import cv2

    cap = cv2.VideoCapture(source)
    debouncer = EventDebouncer()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ids = recognizer.identify(frame)
            top = _top_identification(ids, require_frontal)
            committed = debouncer.update(top.match.name if top else None)
            if committed:
                if on_event:
                    on_event(committed)
                if socket is not None:
                    socket.send(json.dumps({"best_match": committed}))

            if show:
                _draw(frame, ids)
                cv2.imshow("AI Security v2.0", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()


def _top_identification(ids, require_frontal: bool) -> Optional[Identification]:
    candidates = [i for i in ids if i.match.is_known]
    if require_frontal:
        candidates = [i for i in candidates if i.detection.is_frontal()]
    if not candidates:
        return None
    return max(candidates, key=lambda i: i.match.score)


def _draw(frame: np.ndarray, ids) -> None:
    import cv2

    for ident in ids:
        x1, y1, x2, y2 = ident.detection.bbox.astype(int)
        known = ident.match.is_known
        color = (0, 200, 0) if known else (0, 0, 255)
        label = f"{ident.name} {ident.match.score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
