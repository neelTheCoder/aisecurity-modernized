"""Measure end-to-end latency of the detect+embed pipeline.

Useful for picking an execution provider and ``det_size`` for a target device
(e.g. tuning for an on-device mobile build vs. a server GPU).

Usage:
    python -m scripts.benchmark --image path/to/face.jpg --runs 50
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from aisecurity import Config, FaceEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=None, help="image to benchmark on")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--model", default="buffalo_l")
    args = parser.parse_args()

    cfg = Config(model_pack=args.model)
    engine = FaceEngine.from_pretrained(cfg)
    print(f"Providers: {cfg.resolved_providers()}")

    if args.image:
        import cv2

        frame = cv2.imread(args.image)
    else:
        from insightface.data import get_image

        frame = get_image("t1")  # bundled sample with faces

    # warm-up (first run pays kernel/graph init cost)
    engine.detect(frame)

    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        dets = engine.detect(frame)
        times.append((time.perf_counter() - start) * 1000.0)

    times = np.array(times)
    print(f"Faces detected: {len(dets)}")
    print(
        f"Latency over {args.runs} runs: "
        f"mean {times.mean():.1f} ms | p50 {np.percentile(times, 50):.1f} ms | "
        f"p95 {np.percentile(times, 95):.1f} ms"
    )


if __name__ == "__main__":
    main()
