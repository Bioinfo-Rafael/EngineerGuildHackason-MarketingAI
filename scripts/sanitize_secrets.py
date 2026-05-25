#!/usr/bin/env python3
"""Remove OAuth tokens and other secrets from URLs in processed clickstream JSON."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

SECRET_MARKERS = (
    "access_token=",
    "id_token=",
    "refresh_token=",
    "ya29.",
    "eyJhbGciOi",  # JWT header prefix in URLs
)


def sanitize_url(url: str) -> str:
    if not url:
        return url
    if not any(m in url for m in SECRET_MARKERS):
        return url

    parsed = urlparse(url.replace("\\u0026", "&"))
    # Drop query and fragment that carry tokens; keep path for ML structure.
    clean = urlunparse((parsed.scheme, parsed.netloc, parsed.path or "/", "", "", ""))
    if "medium.com/m/callback/google" in clean:
        return "https://medium.com/m/callback/google"
    return clean


def sanitize_obj(obj: dict) -> bool:
    changed = False
    for key in ("previous_url", "current_url"):
        if key not in obj:
            continue
        new = sanitize_url(obj[key])
        if new != obj[key]:
            obj[key] = new
            changed = True
    return changed


def main() -> None:
    total = 0
    for path in sorted(PROCESSED.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        file_changed = False
        for task in data:
            for click in task.get("clickstream", []):
                if sanitize_obj(click):
                    file_changed = True
                    total += 1
        if file_changed:
            with path.open("w") as f:
                json.dump(data, f, indent=4)
                f.write("\n")
            print(f"Updated {path.name}")
    print(f"Sanitized {total} click records across {PROCESSED}")


if __name__ == "__main__":
    main()
