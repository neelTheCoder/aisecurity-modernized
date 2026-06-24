"""Build an encrypted identity gallery from a directory of face images.

Usage:
    python -m scripts.enroll --images data/students --out gallery.aisec

Each file should be one person's face, named ``firstname-lastname.jpg``. Extra
shots use a numeric suffix (``firstname-lastname-2.jpg``) and are merged.

The passphrase is read from the ``AISEC_PASSPHRASE`` environment variable so it
never lands in your shell history.
"""

from __future__ import annotations

import argparse
import getpass
import os

from aisecurity import Config, Recognizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="directory of face images")
    parser.add_argument("--out", default="gallery.aisec", help="output gallery path")
    parser.add_argument("--threshold", type=float, default=0.40, help="match cutoff")
    parser.add_argument("--model", default="buffalo_l", help="InsightFace model pack")
    args = parser.parse_args()

    passphrase = os.environ.get("AISEC_PASSPHRASE") or getpass.getpass(
        "Gallery passphrase: "
    )

    rec = Recognizer.from_pretrained(Config(model_pack=args.model))
    rec.gallery.match_threshold = args.threshold

    report = rec.enroll_dir(args.images)
    print(f"Enrolled {len(report['enrolled'])} image(s) "
          f"across {len(rec.gallery)} identities.")
    if report["skipped"]:
        print(f"Skipped (no face found): {', '.join(report['skipped'])}")

    rec.gallery.save(args.out, passphrase)
    print(f"Encrypted gallery written to {args.out}")


if __name__ == "__main__":
    main()
