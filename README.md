<!-- Ported from millburnai/aisecurity. -->
# AI Security — Facial Recognition Kiosk (2026 refactor)

Privacy-preserving facial recognition for monitoring entry/exit at a school
kiosk. Originally built as a student-led project on the Millburn High School AI
team; this is a ground-up modernization of the computer-vision core to current
(2026) state-of-the-art models and tooling.

A camera at the door detects faces, matches each one against an encrypted
gallery of enrolled students, and logs who is entering or leaving — **without
ever storing a photo**. Only irreversible face *embeddings* are kept, and those
are encrypted at rest.

```python
from aisecurity import Recognizer

rec = Recognizer.from_pretrained()          # SCRFD + ArcFace, ONNX
rec.gallery.enroll_dir("data/students")     # one firstname-lastname.jpg per student
for ident in rec.identify(frame):           # frame = BGR numpy array
    print(ident.name, ident.match.score)
```

---

## System

| Component | Original (≈2021) | This version (2026) | Why |
|---|---|---|---|
| Face detector | MTCNN (TF1 frozen graph) | **SCRFD** | Faster and more accurate, esp. on small/angled faces; single ONNX model. |
| Embeddings | **FaceNet** (2015) | **ArcFace** `w600k_r50` | Angular-margin loss → far better identity separation; current industry standard. |
| Matcher | Linear **SVM / KNN** over embeddings | **Cosine nearest-centroid** | Open-set by design; add/remove a student with no retraining; handles unknown intruders. |
| Runtime | TF1 `compat.v1` + TensorRT (Jetson-only) | **ONNX Runtime** | One model file runs on CPU, CUDA, **CoreML (iOS/Apple)**, or NNAPI (Android). |
| Encryption | AES (unauthenticated), `pycryptodome` | **AES-256-GCM + scrypt**, `cryptography` | Authenticated (tamper-detecting); passphrase-derived key, no key blob on disk. |
| Packaging | pinned `requirements.txt`, no tests | `pyproject.toml`, unit tests, CI, type hints | Reproducible and maintainable. |

---

## Architecture

```
              ┌──────────── FaceEngine (ONNX Runtime) ────────────┐
  BGR frame → │  SCRFD detect → 5-pt landmarks → align → ArcFace  │ → 512-d embedding
              └────────────────────────────────────────────────────┘
                                       │
                                       ▼
                         Gallery.match()  ── cosine vs. encrypted
                                       │      enrolled centroids
                                       ▼
                  Identification(name | "unknown", score)
                                       │
                         EventDebouncer (temporal smoothing)
                                       │
                          → kiosk websocket  {"best_match": name}
```

- [`aisecurity/engine.py`](aisecurity/engine.py) — detection + embedding (InsightFace/ONNX).
- [`aisecurity/gallery.py`](aisecurity/gallery.py) — enrolled identities, cosine matching, encrypted persistence.
- [`aisecurity/recognizer.py`](aisecurity/recognizer.py) — the high-level `identify()` / `enroll` API.
- [`aisecurity/crypto.py`](aisecurity/crypto.py) — AES-256-GCM gallery encryption.
- [`aisecurity/stream.py`](aisecurity/stream.py) — real-time loop + event de-bouncing + kiosk hook.

---

## Privacy model

Three properties, carried over from the original project and strengthened:

1. **No images stored.** Faces are converted to 512-d embeddings and the frame
   is discarded. Embeddings are one-way — the original photo cannot be
   reconstructed from them.
2. **Encrypted at rest.** The gallery is sealed with AES-256-GCM using a key
   derived from a passphrase via scrypt (random salt per file). A stolen
   gallery file is useless without the passphrase, and any tampering is
   detected on load.
3. **Local-only.** Everything runs on-device; nothing is sent to a third party.

---

## Measured behaviour

Verified end-to-end on the bundled InsightFace sample faces (Apple Silicon,
`CoreMLExecutionProvider` + CPU):

- **Identity separation** — genuine self-match cosine ≈ `1.00`; different people
  peak at `0.21`. The default decision threshold of `0.40` sits cleanly in the
  gap, and a random/intruder embedding scores ≈ `0.10` (correctly rejected).
- **Latency** — ≈ **110 ms** to detect + embed a 6-face frame end-to-end on CPU/CoreML
  (single-face frames are a fraction of that; a CUDA/Jetson GPU is faster still).

Reproduce with `python -m scripts.benchmark`.

---

## Install & run

```bash
pip install -r requirements.txt          # CPU / Apple Silicon
# For an NVIDIA GPU/Jetson: pip install onnxruntime-gpu instead of onnxruntime

# 1. Enroll students from a folder of firstname-lastname.jpg photos
export AISEC_PASSPHRASE="choose-a-strong-passphrase"
python -m scripts.enroll --images data/students --out gallery.aisec

# 2. Run live recognition from a webcam
python -m scripts.recognize --gallery gallery.aisec

# 3. (kiosk) stream events to the Django kiosk server, like the original Jetson Nano
python -m scripts.recognize --gallery gallery.aisec --kiosk ws://server:8000/v1/nano
```

The first run downloads the `buffalo_l` model pack (~300 MB) to `~/.insightface`.

## Tests

```bash
pip install pytest && pytest      # 19 tests: crypto, gallery matching, pipeline logic
```

The crypto and matching tests use synthetic vectors, so they run in seconds
with no model download (this is what CI runs).

## Kiosk integration

The original deployment paired this recognizer (on an NVIDIA Jetson Nano) with a
Django + Channels server that logged entries. `scripts/recognize.py --kiosk`
sends the identical `{"best_match": <name>}` websocket message, so it drops into
that existing server unchanged.

---

*License: MIT. Built on [InsightFace](https://github.com/deepinsight/insightface)
(SCRFD + ArcFace) and ONNX Runtime.*
