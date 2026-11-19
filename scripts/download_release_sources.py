"""Download only the versioned, SHA-256-pinned sources listed by the project."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    target = ROOT / ".workbench" / "third-party-sources"
    target.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / "packaging/third-party-sources.json").read_text())
    for source in sources:
        path = target / source["filename"]
        if (
            path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
        ):
            print(f"Verified cached {path.name}", flush=True)
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {source['url']}", flush=True)
        with urllib.request.urlopen(source["url"], timeout=120) as response:
            with temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != source["sha256"]:
            temporary.unlink()
            raise ValueError(f"Source checksum mismatch: {path.name}")
        temporary.replace(path)


if __name__ == "__main__":
    main()
