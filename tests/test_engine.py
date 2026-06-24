# Ported from millburnai/aisecurity.
"""Tests for engine/recognizer logic using a fake model app (no downloads)."""

import numpy as np

from aisecurity import Config, FaceEngine, Recognizer
from aisecurity.stream import EventDebouncer


class _FakeFace:
    def __init__(self, embedding, bbox, kps, score):
        self.normed_embedding = embedding
        self.bbox = bbox
        self.kps = kps
        self.det_score = score


class _FakeApp:
    """Stand-in for insightface FaceAnalysis returning scripted detections."""

    def __init__(self, faces):
        self._faces = faces

    def get(self, frame):
        return self._faces


def _face(embedding, size=200, score=0.99):
    bbox = np.array([100, 100, 100 + size, 100 + size], dtype=np.float32)
    # eyes far apart relative to width => frontal
    kps = np.array(
        [[140, 160], [260, 160], [200, 200], [150, 250], [250, 250]],
        dtype=np.float32,
    )
    return _FakeFace(np.asarray(embedding, dtype=np.float32), bbox, kps, score)


def test_min_face_filter():
    cfg = Config(min_face_px=100)
    big = _face([1, 0, 0], size=200)
    small = _face([0, 1, 0], size=40)
    engine = FaceEngine(cfg, app=_FakeApp([big, small]))
    dets = engine.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(dets) == 1  # small face filtered out


def test_largest_face():
    engine = FaceEngine(Config(min_face_px=10), app=_FakeApp([_face([1, 0, 0], 80), _face([0, 1, 0], 200)]))
    det = engine.largest_face(np.zeros((10, 10, 3), dtype=np.uint8))
    assert det.width == 200


def test_recognizer_enroll_and_identify():
    alice = np.array([1, 0, 0], dtype=np.float32)
    engine = FaceEngine(Config(min_face_px=10), app=_FakeApp([_face(alice)]))
    rec = Recognizer(engine)
    rec.gallery.match_threshold = 0.4

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert rec.enroll_image("alice", frame) is True

    ids = rec.identify(frame)
    assert len(ids) == 1
    assert ids[0].name == "alice"


def test_is_frontal():
    engine = FaceEngine(Config(min_face_px=10), app=_FakeApp([_face([1, 0, 0])]))
    det = engine.detect(np.zeros((10, 10, 3), dtype=np.uint8))[0]
    assert det.is_frontal()


def test_debouncer_commits_after_min_hits():
    d = EventDebouncer(window=5, min_hits=3, cooldown_s=0.0)
    assert d.update("alice") is None
    assert d.update("alice") is None
    assert d.update("alice") == "alice"  # third consecutive hit commits


def test_debouncer_ignores_none():
    d = EventDebouncer(window=5, min_hits=2, cooldown_s=0.0)
    assert d.update(None) is None
    assert d.update("bob") is None
    assert d.update("bob") == "bob"
