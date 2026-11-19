"""Release evidence must exclude local data and reject unsafe archive paths."""

import importlib.util
import io
import tarfile
from pathlib import Path


def release_module():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_release.py"
    spec = importlib.util.spec_from_file_location("prepare_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_snapshot_selects_build_inputs_and_excludes_user_data(tmp_path):
    for name in (
        "src/package/example.json",
        "tests/fixtures/public.csv",
        "uv.lock",
        ".workbench/private.json",
        "local.sqlite3",
        ".env",
        "src/__pycache__/code.pyc",
        "src/package.egg-info/SOURCES.txt",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sample")
    selected = {
        p.relative_to(tmp_path).as_posix()
        for p in release_module().source_files(tmp_path)
    }
    assert selected == {
        "src/package/example.json",
        "tests/fixtures/public.csv",
        "uv.lock",
    }


def test_notice_copy_never_extracts_parent_or_windows_drive_paths(tmp_path):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        for name in (
            "package/LICENSE",
            "package/LICENSES/LGPL-3.0-only.txt",
            "../LICENSE",
            "/LICENSE",
            "C:/LICENSE",
            "a\\LICENSE",
        ):
            member = tarfile.TarInfo(name)
            member.size = 7
            target.addfile(member, io.BytesIO(b"license"))
        link = tarfile.TarInfo("package/COPYING")
        link.type = tarfile.SYMTYPE
        link.linkname = "../outside"
        target.addfile(link)
    output = tmp_path / "notices"
    release_module().copy_source_notices(archive, output)
    assert sorted([
        p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
    ]) == ["package/LICENSE", "package/LICENSES/LGPL-3.0-only.txt"]
