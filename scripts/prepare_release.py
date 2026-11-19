"""Prepare offline release evidence, notices and the exact application sources."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def source_files(root: Path) -> list[Path]:
    """Include scientific/code inputs; never include local databases or credentials."""
    files = [
        root / name
        for name in (
            "LICENSE",
            "README.md",
            "CONTRIBUTING.md",
            "THIRD_PARTY_NOTICES.md",
            "pyproject.toml",
            "uv.lock",
            "MANIFEST.in",
            "conftest.py",
            "bioeng-qpcr.spec",
            "CITATION.cff",
        )
    ]
    for directory in (
        "src",
        "tests",
        "scripts",
        "docs",
        "examples",
        "packaging",
        ".github",
    ):
        files.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in {
                ".py",
                ".toml",
                ".lock",
                ".md",
                ".txt",
                ".json",
                ".html",
                ".csv",
                ".tsv",
                ".xml",
                ".pdb",
                ".fasta",
                ".fastq",
                ".ppm",
                ".ps1",
                ".yaml",
                ".yml",
                ".svg",
                ".png",
            }
            and "__pycache__" not in path.parts
            and not any(part.endswith(".egg-info") for part in path.parts)
        )
    return sorted(set(path for path in files if path.is_file()))


def is_notice(name: str) -> bool:
    path = PurePosixPath(name.lower())
    return "licenses" in path.parts or path.name.startswith(
        ("license", "licence", "copying", "copyright", "notice")
    )


def copy_source_notices(archive: Path, target: Path) -> None:
    """Read regular notice files without extracting paths or links from an archive."""
    with tarfile.open(archive) as source:
        for member in source:
            parts = PurePosixPath(member.name).parts
            if (
                not member.isfile()
                or not is_notice(member.name)
                or PurePosixPath(member.name).is_absolute()
                or ".." in parts
                or any(":" in part or "\\" in part for part in parts)
            ):
                continue
            stream = source.extractfile(member)
            if stream:
                path = target.joinpath(*parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as output:
                    shutil.copyfileobj(stream, output)


def prepare_release(root: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    licenses = output / "licenses"
    licenses.mkdir(exist_ok=True)
    inventory = []
    for dist in sorted(
        importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower()
    ):
        name = dist.metadata["Name"]
        destination = licenses / f"{name}-{dist.version}"
        destination.mkdir(exist_ok=True)
        notices = []
        for entry in dist.files or []:
            if is_notice(str(entry)) and Path(dist.locate_file(entry)).is_file():
                safe = [
                    part
                    for part in PurePosixPath(str(entry)).parts
                    if part not in ("..", "/")
                ]
                path = destination.joinpath(*safe)
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(dist.locate_file(entry), path)
                notices.append(path.relative_to(licenses).as_posix())
        # Some wheels carry the full permission/copyright text only in METADATA.
        metadata = dist.read_text("METADATA") or dist.read_text("PKG-INFO") or ""
        (destination / "METADATA.txt").write_text(metadata, encoding="utf-8")
        inventory.append(
            {
                "name": name,
                "version": dist.version,
                "license_expression": dist.metadata.get("License-Expression"),
                "license": dist.metadata.get("License"),
                "notices": notices,
            }
        )
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if not python_license.is_file():
        raise ValueError("Python runtime LICENSE.txt is missing")
    shutil.copyfile(python_license, licenses / "PYTHON-LICENSE.txt")
    sources = json.loads((root / "packaging/third-party-sources.json").read_text())
    for source in sources:
        path = root / ".workbench/third-party-sources" / source["filename"]
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]
        ):
            raise ValueError(
                f"Missing/changed source: {path.name}; "
                "run scripts/download_release_sources.py"
            )
        if source.get("distribution"):
            actual = importlib.metadata.version(source["distribution"])
            if actual != source["version"]:
                raise ValueError(
                    f"Source {source['version']} does not match "
                    f"{source['distribution']} {actual}"
                )
        copy_source_notices(path, licenses / "source-notices")
    (licenses / "DISTRIBUTIONS.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    from bioeng_qpcr import __version__

    paths = source_files(root)
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    source_hash = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    info = {
        "software_version": __version__,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git.stdout.strip() if git.returncode == 0 else None,
        "git_dirty": bool(status.stdout) if status.returncode == 0 else None,
        "source_tree_sha256": source_hash,
        "source_files": hashes,
        "python": sys.version,
        "dependencies": {d["name"]: d["version"] for d in inventory},
        "third_party_sources": sources,
    }
    # Keep the large file inventory in the release, not in every analysis record.
    build_path = output / "build-info.json"
    build_path.write_text(
        json.dumps({k: v for k, v in info.items() if k != "source_files"}, indent=2),
        encoding="utf-8",
    )
    (output / "BUILD_INFO.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    with zipfile.ZipFile(
        output / "bioeng-qpcr-source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
    return output
