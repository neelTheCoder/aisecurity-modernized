"""Runtime configuration for the recognition pipeline.

Everything tunable lives here so that detector/embedder/matcher behaviour can be
adjusted (or pinned for reproducibility) without touching pipeline code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# InsightFace model packs. ``buffalo_l`` = SCRFD-10GF detector + ArcFace
# ``w600k_r50`` (ResNet-50, 512-d) embeddings. ``buffalo_sc`` is a lighter
# pack better suited to constrained edge devices.
DEFAULT_MODEL_PACK = "buffalo_l"


def default_providers() -> List[str]:
    """Pick the best available ONNX Runtime execution provider.

    The model is provider-agnostic, so the same ``.onnx`` files run on:

    * ``CUDAExecutionProvider``  — server / Jetson GPU
    * ``CoreMLExecutionProvider`` — Apple Silicon and iOS (relevant for a
      phone-based setup app)
    * ``CPUExecutionProvider``   — universal fallback

    We probe what is installed at runtime and order by preference; ONNX Runtime
    falls through the list until one initialises.
    """
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:  # pragma: no cover - onnxruntime always present in prod
        return ["CPUExecutionProvider"]

    preference = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    ordered = [p for p in preference if p in available]
    return ordered or ["CPUExecutionProvider"]


@dataclass
class Config:
    """Pipeline configuration.

    :param model_pack: InsightFace model pack name.
    :param det_size: detector input resolution (w, h). Larger = more accurate
        on small/distant faces, slower.
    :param det_threshold: minimum SCRFD detection confidence to keep a face.
    :param match_threshold: cosine-similarity cutoff for a positive identity
        match. ArcFace embeddings are L2-normalised, so this is in [-1, 1].
        ~0.40 is a sensible default for ``w600k_r50``; calibrate per camera.
    :param min_face_px: ignore detections whose bounding box is smaller than
        this (filters out background faces / false positives).
    :param providers: ONNX Runtime execution providers (auto-detected if None).
    :param ctx_id: device index for GPU providers (-1 forces CPU).
    """

    model_pack: str = DEFAULT_MODEL_PACK
    det_size: tuple[int, int] = (640, 640)
    det_threshold: float = 0.5
    match_threshold: float = 0.40
    min_face_px: int = 60
    providers: Optional[List[str]] = field(default=None)
    ctx_id: int = 0

    def resolved_providers(self) -> List[str]:
        return self.providers if self.providers is not None else default_providers()
