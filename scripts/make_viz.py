#!/usr/bin/env python3
"""Build the standalone visualisation page.

Combines ``scripts/viz_template.html`` with a run exported by
``scripts/export_sim.py`` into a single self-contained file under ``viz/``.
No network, no CDN, no build step: the data is inlined, so the result opens
from the filesystem and keeps working with no connection.

    python3 scripts/export_sim.py --scenario drone --run 0 --stride 2
    python3 scripts/make_viz.py
    python3 scripts/make_viz.py --serve      # and serve it on localhost

Regenerating from committed results is the point. A figure that cannot be
rebuilt from the data is a figure nobody can check.
"""

from __future__ import annotations

import argparse
import http.server
import json
import pathlib
import socketserver
import sys
import webbrowser

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLACEHOLDER = "__DATA__"


def build(data_path: pathlib.Path, out: pathlib.Path,
          template: pathlib.Path) -> pathlib.Path:
    if not data_path.exists():
        sys.exit(
            f"{data_path} not found. Run:\n"
            f"  python3 scripts/export_sim.py --scenario drone --run 0 --stride 2"
        )
    if not template.exists():
        sys.exit(f"{template} not found")
    tpl = template.read_text()
    n = tpl.count(PLACEHOLDER)
    if n != 1:
        sys.exit(
            f"{template} contains the placeholder {n} times; it must be exactly one. "
            "More than one means the data block gets substituted somewhere else too, "
            "which silently breaks the built page."
        )

    data = data_path.read_text()
    json.loads(data)  # fail here rather than silently in the browser

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tpl.replace(PLACEHOLDER, data))
    return out


def serve(directory: pathlib.Path, page: str, port: int, open_browser: bool) -> None:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

        def log_message(self, *a):  # keep the console quiet
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/{page}"
        print(f"serving {directory} at {url}")
        print("ctrl-c to stop")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, default=ROOT / "results" / "sim.json")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "viz" / "drone_sim.html")
    ap.add_argument("--template", type=pathlib.Path,
                    default=ROOT / "scripts" / "viz_template.html",
                    help="viz_template.html for one run, viz_compare.html for the case set")
    ap.add_argument("--pages", action="store_true",
                    help="also write docs/index.html, which is what GitHub Pages serves")
    ap.add_argument("--serve", action="store_true", help="serve on localhost after building")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    a = ap.parse_args()

    out = build(a.data, a.out, a.template)
    kb = out.stat().st_size / 1024
    print(f"wrote {out} ({kb:.0f} KB, self-contained)")
    print(f"open directly:  file://{out}")

    if a.pages:
        pages = ROOT / "docs" / "index.html"
        pages.parent.mkdir(exist_ok=True)
        pages.write_text(out.read_text())
        print(f"wrote {pages} (GitHub Pages entry point)")

    if a.serve:
        serve(out.parent, out.name, a.port, not a.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
