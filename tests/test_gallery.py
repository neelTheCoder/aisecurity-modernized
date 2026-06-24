# Ported from millburnai/aisecurity.
"""Tests for gallery matching and encrypted persistence.

These use synthetic embeddings, so they run fast and need no model download.
"""

import numpy as np
import pytest

from aisecurity.gallery import Gallery


def _unit(*vals) -> np.ndarray:
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_exact_match():
    g = Gallery(match_threshold=0.4)
    alice = _unit(1, 0, 0, 0)
    g.add("alice", alice)
    result = g.match(alice)
    assert result.name == "alice"
    assert result.is_known
    assert result.score == pytest.approx(1.0, abs=1e-5)


def test_intruder_below_threshold():
    g = Gallery(match_threshold=0.4)
    g.add("alice", _unit(1, 0, 0, 0))
    result = g.match(_unit(0, 1, 0, 0))  # orthogonal => cosine 0
    assert result.name is None
    assert result.is_intruder


def test_centroid_of_multiple_shots():
    g = Gallery(match_threshold=0.4)
    g.add("bob", _unit(1, 0.1, 0, 0))
    g.add("bob", _unit(1, -0.1, 0, 0))
    # a probe near the average of bob's shots should match bob
    result = g.match(_unit(1, 0, 0, 0))
    assert result.name == "bob"


def test_picks_nearest_identity():
    g = Gallery(match_threshold=0.3)
    g.add("alice", _unit(1, 0, 0))
    g.add("bob", _unit(0, 1, 0))
    probe = _unit(0.9, 0.2, 0)
    assert g.match(probe).name == "alice"


def test_remove():
    g = Gallery()
    g.add("alice", _unit(1, 0, 0))
    g.remove("alice")
    assert len(g) == 0
    assert g.match(_unit(1, 0, 0)).name is None


def test_empty_gallery_returns_unknown():
    assert Gallery().match(_unit(1, 0, 0)).name is None


def test_serialize_roundtrip():
    g = Gallery(match_threshold=0.37)
    g.add("alice", _unit(1, 0, 0, 0))
    g.add("bob", _unit(0, 1, 0, 0))
    g.add("bob", _unit(0, 0.9, 0.1, 0))

    restored = Gallery.deserialize(g.serialize())
    assert sorted(restored.names) == ["alice", "bob"]
    assert restored.match_threshold == pytest.approx(0.37)
    assert restored.match(_unit(1, 0, 0, 0)).name == "alice"


def test_encrypted_save_load(tmp_path):
    g = Gallery()
    g.add("alice", _unit(1, 0, 0, 0))
    path = tmp_path / "g.aisec"
    g.save(str(path), "pw123")

    loaded = Gallery.load(str(path), "pw123")
    assert loaded.match(_unit(1, 0, 0, 0)).name == "alice"

    # file on disk must not contain the plaintext name
    assert b"alice" not in path.read_bytes()
