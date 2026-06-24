# Ported from millburnai/aisecurity.
"""AI Security — modern, privacy-preserving facial recognition.

A 2026 refactor of the original Millburn AI ``aisecurity`` project. The core
models have been replaced with current state-of-the-art, ONNX-based components:

* **Detection** — SCRFD (replaces MTCNN)
* **Embeddings** — ArcFace ``w600k_r50`` (replaces FaceNet, 2015)
* **Runtime** — ONNX Runtime, with CoreML / NNAPI / CUDA execution providers
  so the exact same model files run on a laptop, a server GPU, or on-device
  in a phone app.

The public surface is intentionally small:

    >>> from aisecurity import Recognizer
    >>> rec = Recognizer.from_pretrained()
    >>> rec.gallery.enroll_dir("data/students")
    >>> result = rec.identify(frame)        # frame is a BGR numpy array
"""

from aisecurity.config import Config
from aisecurity.engine import FaceEngine, Detection
from aisecurity.gallery import Gallery, Match
from aisecurity.recognizer import Recognizer

__all__ = [
    "Config",
    "FaceEngine",
    "Detection",
    "Gallery",
    "Match",
    "Recognizer",
]

__version__ = "2.0.0"
