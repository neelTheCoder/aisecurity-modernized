"""High-level recognizer: ties the model engine to the identity gallery.

This is the object most callers use. It mirrors the role of the original
``FaceNet`` class but with a much smaller, clearer surface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from aisecurity.config import Config
from aisecurity.engine import Detection, FaceEngine
from aisecurity.gallery import Gallery, Match


@dataclass
class Identification:
    """A detected face plus its identity decision."""

    detection: Detection
    match: Match

    @property
    def name(self) -> str:
        return self.match.name or "unknown"


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class Recognizer:
    """End-to-end: frame in, identified faces out."""

    def __init__(self, engine: FaceEngine, gallery: Optional[Gallery] = None):
        self.engine = engine
        self.gallery = gallery or Gallery(match_threshold=engine.config.match_threshold)

    @classmethod
    def from_pretrained(cls, config: Config | None = None) -> "Recognizer":
        engine = FaceEngine.from_pretrained(config)
        return cls(engine)

    # -- inference ----------------------------------------------------------

    def identify(self, frame_bgr: np.ndarray) -> List[Identification]:
        """Detect and identify every face in a BGR frame."""
        results: List[Identification] = []
        for det in self.engine.detect(frame_bgr):
            results.append(Identification(det, self.gallery.match(det.embedding)))
        return results

    def identify_one(self, frame_bgr: np.ndarray) -> Optional[Identification]:
        """Identify only the largest (closest) face — kiosk/enrollment use."""
        det = self.engine.largest_face(frame_bgr)
        if det is None:
            return None
        return Identification(det, self.gallery.match(det.embedding))

    # -- enrollment ---------------------------------------------------------

    def enroll_image(self, name: str, frame_bgr: np.ndarray) -> bool:
        """Add one photo of ``name`` to the gallery. Returns False if no face."""
        det = self.engine.largest_face(frame_bgr)
        if det is None:
            return False
        self.gallery.add(name, det.embedding)
        return True

    def enroll_dir(self, img_dir: str) -> dict:
        """Enroll a directory of ``name.jpg`` images (one face per file).

        Files named ``john-smith.jpg`` and ``john-smith-2.jpg`` are merged into
        the same identity (the ``-<n>`` suffix is treated as a sample index).
        Returns ``{"enrolled": [...], "skipped": [...]}``.
        """
        import cv2

        enrolled, skipped = [], []
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(_IMG_EXTS):
                continue
            stem = os.path.splitext(fname)[0]
            name = stem.rsplit("-", 1)[0] if stem.rsplit("-", 1)[-1].isdigit() else stem
            img = cv2.imread(os.path.join(img_dir, fname))
            if img is not None and self.enroll_image(name, img):
                enrolled.append(fname)
            else:
                skipped.append(fname)
        return {"enrolled": enrolled, "skipped": skipped}
