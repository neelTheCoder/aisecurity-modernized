"""Face detection + embedding, wrapped around InsightFace / ONNX Runtime.

This module is the modern replacement for the original ``facenet.py`` +
``util/detection.py``. Instead of a TensorFlow-1 frozen-graph FaceNet and a
separate MTCNN graph, a single InsightFace ``FaceAnalysis`` app runs both
stages as ONNX models:

    SCRFD detector  ->  5-point landmarks  ->  similarity-transform align
                    ->  ArcFace embedder   ->  512-d L2-normalised vector

Embeddings are L2-normalised, so identity comparison is a plain dot product
(cosine similarity) — see :mod:`aisecurity.gallery`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from aisecurity.config import Config


@dataclass
class Detection:
    """One detected face in a frame.

    :param embedding: 512-d, L2-normalised ArcFace embedding.
    :param bbox: (x1, y1, x2, y2) in pixel coordinates.
    :param landmarks: 5x2 facial keypoints (eyes, nose, mouth corners).
    :param det_score: SCRFD detection confidence in [0, 1].
    """

    embedding: np.ndarray
    bbox: np.ndarray
    landmarks: np.ndarray
    det_score: float

    @property
    def width(self) -> int:
        return int(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return int(self.bbox[3] - self.bbox[1])

    def is_frontal(self, min_eye_ratio: float = 0.22) -> bool:
        """Cheap head-pose gate: are both eyes far enough apart to be looking
        roughly at the camera? Mirrors the original ``is_looking`` heuristic,
        but uses ArcFace's stable 5-point landmarks.
        """
        left_eye, right_eye = self.landmarks[0], self.landmarks[1]
        eye_dist = abs(right_eye[0] - left_eye[0])
        return self.width > 0 and (eye_dist / self.width) > min_eye_ratio


class FaceEngine:
    """Thin wrapper over an InsightFace ``FaceAnalysis`` app."""

    def __init__(self, config: Config | None = None, app=None):
        self.config = config or Config()
        self._app = app  # injectable for testing without model downloads

    @classmethod
    def from_pretrained(cls, config: Config | None = None) -> "FaceEngine":
        """Load the configured model pack, downloading it on first use."""
        from insightface.app import FaceAnalysis

        cfg = config or Config()
        app = FaceAnalysis(name=cfg.model_pack, providers=cfg.resolved_providers())
        app.prepare(ctx_id=cfg.ctx_id, det_size=cfg.det_size, det_thresh=cfg.det_threshold)
        return cls(cfg, app)

    @property
    def embedding_dim(self) -> int:
        return 512

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Detect, align, and embed every usable face in a BGR frame."""
        if self._app is None:
            raise RuntimeError(
                "FaceEngine has no model loaded; use FaceEngine.from_pretrained()"
            )

        faces = self._app.get(frame_bgr)
        detections: List[Detection] = []
        for f in faces:
            det = Detection(
                embedding=np.asarray(f.normed_embedding, dtype=np.float32),
                bbox=np.asarray(f.bbox, dtype=np.float32),
                landmarks=np.asarray(f.kps, dtype=np.float32),
                det_score=float(f.det_score),
            )
            if min(det.width, det.height) >= self.config.min_face_px:
                detections.append(det)
        return detections

    def largest_face(self, frame_bgr: np.ndarray) -> Detection | None:
        """Convenience for enrollment/kiosk use: the single closest face."""
        dets = self.detect(frame_bgr)
        if not dets:
            return None
        return max(dets, key=lambda d: d.width * d.height)
