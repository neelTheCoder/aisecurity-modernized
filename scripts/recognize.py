# Ported from millburnai/aisecurity.
"""Run real-time facial recognition against an encrypted gallery.

Usage:
    python -m scripts.recognize --gallery gallery.aisec
    python -m scripts.recognize --gallery gallery.aisec --kiosk ws://server:8000/v1/nano

With ``--kiosk`` it streams ``{"best_match": name}`` events to the existing
Django kiosk server, exactly like the original Jetson Nano client.
"""

from __future__ import annotations

import argparse
import getpass
import os

from aisecurity import Config, Recognizer
from aisecurity.gallery import Gallery
from aisecurity.stream import stream


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", required=True, help="encrypted gallery file")
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--kiosk", default=None, help="kiosk websocket URL")
    parser.add_argument("--no-window", action="store_true", help="headless mode")
    parser.add_argument("--model", default="buffalo_l", help="InsightFace model pack")
    args = parser.parse_args()

    passphrase = os.environ.get("AISEC_PASSPHRASE") or getpass.getpass(
        "Gallery passphrase: "
    )

    rec = Recognizer.from_pretrained(Config(model_pack=args.model))
    rec.gallery = Gallery.load(args.gallery, passphrase)
    print(f"Loaded {len(rec.gallery)} identities.")

    socket = None
    if args.kiosk:
        import websocket  # lazy import; only needed for kiosk mode

        socket = websocket.create_connection(args.kiosk)
        socket.send('{"id": "1"}')

    source = int(args.source) if args.source.isdigit() else args.source
    stream(
        rec,
        source=source,
        show=not args.no_window,
        socket=socket,
        on_event=lambda name: print(f"[event] {name}"),
    )


if __name__ == "__main__":
    main()
