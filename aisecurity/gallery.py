# Ported from millburnai/aisecurity.
"""The identity gallery: enrolled people and nearest-neighbour matching.

Design note — *why not an SVM/KNN classifier?*  The original project trained a
linear SVM over FaceNet embeddings. Modern face recognition treats this as an
open-set retrieval problem instead: ArcFace is trained with an angular-margin
loss specifically so that **cosine similarity between L2-normalised embeddings**
is the identity metric. A direct nearest-centroid match against the gallery is
therefore both simpler and stronger than a learned classifier — and, crucially,
it handles unknown people (intruders) and lets you add/remove a student without
retraining anything.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from aisecurity import crypto


@dataclass
class Match:
    """Result of comparing a probe embedding against the gallery."""

    name: Optional[str]  # None => no enrolled identity cleared the threshold
    score: float  # cosine similarity of the best candidate
    is_known: bool

    @property
    def is_intruder(self) -> bool:
        return not self.is_known


class Gallery:
    """An in-memory, optionally-encrypted store of identity embeddings.

    Each person maps to one or more L2-normalised embeddings. Matching scores a
    probe against every person's **centroid** (the averaged, renormalised
    embedding), which is robust to a few noisy enrollment shots.
    """

    def __init__(self, match_threshold: float = 0.40):
        self.match_threshold = match_threshold
        self._embeddings: Dict[str, List[np.ndarray]] = {}
        self._centroids: Dict[str, np.ndarray] = {}

    # -- enrollment ---------------------------------------------------------

    def add(self, name: str, embedding: np.ndarray) -> None:
        """Enroll a single embedding under ``name`` (idempotent-friendly)."""
        vec = _l2_normalize(np.asarray(embedding, dtype=np.float32).ravel())
        self._embeddings.setdefault(name, []).append(vec)
        self._recompute_centroid(name)

    def remove(self, name: str) -> None:
        self._embeddings.pop(name, None)
        self._centroids.pop(name, None)

    def _recompute_centroid(self, name: str) -> None:
        stacked = np.stack(self._embeddings[name], axis=0)
        self._centroids[name] = _l2_normalize(stacked.mean(axis=0))

    @property
    def names(self) -> List[str]:
        return list(self._embeddings.keys())

    def __len__(self) -> int:
        return len(self._embeddings)

    # -- matching -----------------------------------------------------------

    def match(self, embedding: np.ndarray) -> Match:
        """Return the best identity for a probe embedding.

        Cosine similarity == dot product here because every vector is
        L2-normalised. A match below ``match_threshold`` is reported as an
        unknown person (intruder) rather than forced onto the nearest identity.
        """
        if not self._centroids:
            return Match(name=None, score=0.0, is_known=False)

        probe = _l2_normalize(np.asarray(embedding, dtype=np.float32).ravel())
        names = list(self._centroids.keys())
        centroids = np.stack([self._centroids[n] for n in names], axis=0)
        sims = centroids @ probe

        best = int(np.argmax(sims))
        best_score = float(sims[best])
        is_known = best_score >= self.match_threshold
        return Match(
            name=names[best] if is_known else None,
            score=best_score,
            is_known=is_known,
        )

    # -- persistence (encrypted at rest) ------------------------------------

    def serialize(self) -> bytes:
        """Pack the gallery to compressed bytes (plaintext, pre-encryption)."""
        buf = io.BytesIO()
        payload = {name: np.stack(v, axis=0) for name, v in self._embeddings.items()}
        meta = json.dumps(
            {"match_threshold": self.match_threshold, "names": self.names}
        ).encode("utf-8")
        np.savez_compressed(buf, __meta__=np.frombuffer(meta, dtype=np.uint8), **payload)
        return buf.getvalue()

    @classmethod
    def deserialize(cls, raw: bytes) -> "Gallery":
        with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
            meta = json.loads(bytes(npz["__meta__"]).decode("utf-8"))
            gallery = cls(match_threshold=meta["match_threshold"])
            for name in meta["names"]:
                for vec in npz[name]:
                    gallery.add(name, vec)
        return gallery

    def save(self, path: str, passphrase: str) -> None:
        """Encrypt and write the gallery to ``path``."""
        blob = crypto.encrypt(self.serialize(), passphrase)
        with open(path, "wb") as f:
            f.write(blob)

    @classmethod
    def load(cls, path: str, passphrase: str) -> "Gallery":
        with open(path, "rb") as f:
            blob = f.read()
        return cls.deserialize(crypto.decrypt(blob, passphrase))


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else (vec / norm).astype(np.float32)
