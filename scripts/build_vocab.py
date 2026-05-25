#!/usr/bin/env python3
"""Regenerate vocabs.txt from processed clickstreams (optional; default uses copied vocabs)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "vocabs.txt"

SPECIALS = [
    "<PAD>",
    "<SOA>",
    "<COI>",
    "<SOP>",
    "<EOA_GOAL>",
    "<EOA_FUZZY>",
    "<EOS_EXPLORE>",
    "<MIS>",
]


def main() -> None:
    urls: set[str] = set()
    for path in sorted(PROCESSED.glob("*.json")):
        with path.open() as f:
            tasks = json.load(f)
        for task in tasks:
            for obj in task["clickstream"]:
                urls.add(obj["previous_url"])
    with OUT.open("w") as f:
        f.write("\n".join(SPECIALS))
        f.write("\n")
        f.write("\n".join(sorted(urls)))
    print(f"Wrote {len(SPECIALS) + len(urls)} tokens to {OUT}")


if __name__ == "__main__":
    main()
