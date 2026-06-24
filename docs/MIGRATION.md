<!-- Ported from millburnai/aisecurity. -->
# Migration notes: 2021 stack → 2026 stack

This document explains the engineering decisions behind the refactor. It's
written so a reviewer can see *why* each change was made, not just *what*
changed.

## 1. FaceNet → ArcFace embeddings

The original recognizer used **FaceNet** (Schroff et al., 2015), trained with a
triplet loss. The model worked, but triplet training is notoriously hard to
sample well, and the resulting embedding space separates identities less
cleanly than modern margin-based losses.

**ArcFace** (Deng et al., 2019) adds an *additive angular margin* to the
softmax, which directly maximises inter-class angular distance on the unit
hypersphere. In practice this means: after L2-normalisation, cosine similarity
between two embeddings is a calibrated identity score. We use the
`w600k_r50` weights (ResNet-50 backbone trained on Glint360K), distributed as
ONNX in InsightFace's `buffalo_l` pack.

Concrete payoff, measured on sample faces: genuine matches sit near `1.0` while
different identities top out around `0.21`, leaving a wide, safe decision
margin. FaceNet's margins on the same task were far tighter.

## 2. MTCNN → SCRFD detection

MTCNN is a three-stage cascade (P-Net/R-Net/O-Net) and, in the original repo,
was run as a TensorFlow-1 frozen graph — slow to load and awkward to maintain.

**SCRFD** (Sample and Computation Redistribution for Face Detection, 2021) is a
single-shot detector that is both faster and more accurate on small and
non-frontal faces — important for a doorway camera where people approach at an
angle. It ships as one ONNX model and returns the 5 facial landmarks ArcFace
needs for alignment, so detection and alignment share one clean path.

## 3. SVM/KNN classifier → cosine nearest-centroid

The original trained a linear SVM (or KNN) over FaceNet embeddings. Two problems:

1. **Closed-set.** An SVM assigns *every* input to one of its trained classes.
   A stranger at the door gets confidently labelled as the nearest student. The
   original worked around this with a separate "intruder"/threshold heuristic
   layered on top.
2. **Retraining.** Adding or removing a student meant refitting the classifier.

Because ArcFace embeddings are already an identity-metric space, we drop the
learned classifier entirely and match a probe against each enrolled person's
**centroid** (mean of their enrollment shots, renormalised) by cosine
similarity. Anything below threshold is reported as `unknown`. Enrolling or
removing a person is now an O(1) dictionary update — no training step.

## 4. TF1 + TensorRT → ONNX Runtime

The original ran FaceNet via `tensorflow.compat.v1` frozen graphs, with a
separate hand-built TensorRT path for the Jetson Nano (custom `.engine` files,
PyCUDA memory management). That meant two code paths and a Jetson-only
deployment.

**ONNX Runtime** runs the same `.onnx` files everywhere and picks an execution
provider at runtime:

- `CUDAExecutionProvider` — server / Jetson GPU
- `CoreMLExecutionProvider` — Apple Silicon and **iOS** (a phone setup app)
- `NNAPIExecutionProvider` — Android
- `CPUExecutionProvider` — universal fallback

`Config.resolved_providers()` probes what's installed and orders by preference.
This is the change most relevant to shipping recognition *inside a mobile app*:
no model conversion per platform, just a different provider flag.

## 5. AES → AES-256-GCM with scrypt

The original encrypted its embedding database with `pycryptodome`. We keep the
"encrypt the gallery" idea but upgrade the primitives:

- **AES-256-GCM** is *authenticated* encryption — a tampered gallery fails to
  decrypt instead of silently returning corrupted data.
- The key is derived from a passphrase with **scrypt** (random per-file salt),
  so there's no raw key file sitting next to the data.
- Built on `cryptography`, which is actively maintained and audited.

See [`aisecurity/crypto.py`](../aisecurity/crypto.py).

## What was intentionally kept

- **The privacy model** (embeddings only, encrypted, local) — still the right
  design, just with stronger crypto.
- **Temporal de-bouncing** — only commit an identity after it's stable across
  several frames. Carried into `stream.EventDebouncer`.
- **The kiosk websocket contract** — `{"best_match": name}` — so this still
  drops into the existing Django `aisecurity_server` deployment.

## Possible next steps

- **Liveness / anti-spoofing.** A camera-facing kiosk should reject photos and
  video replays; a lightweight on-device liveness model (e.g. Silent-Face) is
  the natural next addition.
- **Threshold calibration per camera.** Ship a small script that sweeps the
  match threshold against a labelled validation set to set the FAR/FRR operating
  point for a specific install.
- **Quantized/INT8 ONNX** for the lowest-power edge or mobile targets.
