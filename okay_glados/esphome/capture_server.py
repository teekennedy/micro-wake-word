#!/usr/bin/env python3
"""Receive tagged audio from the audio_capture ESPHome component.

Run it on any machine on the LAN (standard library only):

    python3 capture_server.py --dir /path/to/captures --port 8787

Each POST becomes  <dir>/<label>/<device>_<YYYYmmdd-HHMMSS>.wav
A POST with an X-Relabel-Last header instead moves that device's newest
capture into another label folder (detected/ -> false_accept/).
"""

import argparse
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BYTES = 8 * 1024 * 1024  # a 60 s buffer is ~1.9 MB
RELABEL_WINDOW_S = 600       # only re-tag a detection from the last 10 minutes
AMBIENT_LABEL = "true_reject"
AMBIENT_DISCARD_S = 30       # an ambient sample this close before another upload is suspect


def safe(value: str, default: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value or "")[:64] or default


class Handler(BaseHTTPRequestHandler):
    out_dir: Path

    def relabel_last(self, label: str):
        """Move this device's newest detection into <label>/ (if it is recent)."""
        device = safe(self.headers.get("X-Device"), "device")
        # Only ever a wake word detection: never a sampled true_reject clip or
        # a clip someone tagged by hand.
        recent = [p for p in (self.out_dir / "detected").glob(f"{device}_*.wav")
                  if time.time() - p.stat().st_mtime < RELABEL_WINDOW_S]
        if not recent:
            self.send_error(404, "no recent capture to re-tag")
            return
        src = max(recent, key=lambda p: p.stat().st_mtime)
        dest = self.out_dir / label / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dest)
        print(f"re-tagged {src} -> {dest}", flush=True)
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if self.headers.get("X-Relabel-Last"):
            self.rfile.read(length)
            self.relabel_last(safe(self.headers.get("X-Relabel-Last"), "unlabelled"))
            return
        if not 44 < length <= MAX_BYTES:
            self.send_error(400, "bad length")
            return
        body = self.rfile.read(length)
        if body[:4] != b"RIFF" or body[8:12] != b"WAVE":
            self.send_error(400, "not a wav file")
            return
        label = safe(self.headers.get("X-Label"), "unlabelled")
        device = safe(self.headers.get("X-Device"), "device")
        folder = self.out_dir / label
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{device}_{time.strftime('%Y%m%d-%H%M%S')}.wav"
        n = 1
        while path.exists():
            path = path.with_name(f"{path.stem.split('.')[0]}.{n}.wav")
            n += 1
        path.write_bytes(body)
        if label != AMBIENT_LABEL:
            # An ambient sample taken just before a detection or a tagged miss
            # probably contains (part of) a wake word: it is not a negative.
            for old in (self.out_dir / AMBIENT_LABEL).glob(f"{device}_*.wav"):
                if time.time() - old.stat().st_mtime < AMBIENT_DISCARD_S:
                    dest = self.out_dir / "discarded" / old.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    old.rename(dest)
                    print(f"discarded {old} (followed by '{label}')", flush=True)
        print(f"saved {path} ({length} bytes)", flush=True)
        self.send_response(204)
        self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("captures"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    Handler.out_dir = args.dir
    args.dir.mkdir(parents=True, exist_ok=True)
    print(f"saving captures to {args.dir.resolve()} on port {args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
