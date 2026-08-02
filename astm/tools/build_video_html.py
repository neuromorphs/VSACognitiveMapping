"""Inline a video_<scene>.json payload into the player template.

Kept separate from the exporter so the page can be restyled without re-running
the (slow) frame export, and so the same template serves every scene.

    python tools/build_video_html.py --scene school_run1
"""
from __future__ import annotations

import argparse
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "video_page_template.html")

TITLES = {
    "classroom": "Classroom walk, replayed — one memory per encoder",
    "school_run1": "School run, replayed — one memory per encoder",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="school_run1")
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = args.json or os.path.join("outputs", args.scene, f"video_{args.scene}.json")
    out = args.out or os.path.join("outputs", args.scene, f"video_{args.scene}.html")

    tpl = io.open(TEMPLATE, encoding="utf-8").read()
    payload = io.open(src, encoding="utf-8").read()

    # `</` would end the <script> block early. In JSON, `/` outside a string is
    # impossible and `\/` inside one is a legal escape for `/`, so this is safe.
    payload = payload.replace("</", "<\\/")

    html = tpl.replace("__PAYLOAD__", payload)
    html = html.replace("__TITLE__", TITLES.get(args.scene, args.scene + " — video results"))

    io.open(out, "w", encoding="utf-8", newline="").write(html)
    print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
