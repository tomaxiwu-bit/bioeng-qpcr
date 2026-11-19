"""SQLite persistence with revision checks, keeping every acquisition snapshot."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from bioeng_qpcr.workbench.model import Project


class RevisionConflict(ValueError):
    """A different client saved this experiment first."""


class ProjectStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=15)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS projects "
            "(id TEXT PRIMARY KEY, name TEXT NOT NULL, revision INTEGER NOT NULL, "
            "updated_at TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS analysis_records "
            "(id TEXT PRIMARY KEY, module TEXT NOT NULL, created_at TEXT NOT NULL, "
            "title TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def list(self) -> list[dict]:
        with self.connect() as db:
            return [
                dict(zip(("id", "name", "revision", "updated_at"), row, strict=True))
                for row in db.execute(
                    "SELECT id,name,revision,updated_at FROM projects "
                    "ORDER BY updated_at DESC"
                )
            ]

    def get(self, project_id: str) -> Project:
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM projects WHERE id=?", (project_id,)
            ).fetchone()
        if row is None:
            raise KeyError("实验项目不存在")
        return Project.from_dict(json.loads(row[0]))

    def save(self, project: Project, expected: int | None = None) -> None:
        revision = project.revision + 1
        payload = asdict(project)
        payload["revision"] = revision
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self.connect() as db:
            if expected is None:
                db.execute(
                    "INSERT INTO projects VALUES (?,?,?,?,?)",
                    (
                        project.id,
                        project.name,
                        revision,
                        project.updated_at,
                        encoded,
                    ),
                )
            else:
                cursor = db.execute(
                    "UPDATE projects SET name=?,revision=?,updated_at=?,payload=? "
                    "WHERE id=? AND revision=?",
                    (
                        project.name,
                        revision,
                        project.updated_at,
                        encoded,
                        project.id,
                        expected,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RevisionConflict(
                        "项目已被另一窗口更新，请刷新后重试；当前修改尚未保存"
                    )
        project.revision = revision

    def save_record(self, record: dict) -> None:
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self.connect() as db:
            db.execute(
                "INSERT INTO analysis_records VALUES (?,?,?,?,?)",
                (
                    record["id"],
                    record["module"],
                    record["created_at"],
                    record["title"],
                    encoded,
                ),
            )

    def list_records(self, module: str | None) -> list[dict]:
        with self.connect() as db:
            return [
                dict(zip(("id", "module", "created_at", "title"), row, strict=True))
                for row in db.execute(
                    "SELECT id,module,created_at,title FROM analysis_records "
                    "WHERE (? IS NULL OR module=?) "
                    "ORDER BY created_at DESC,rowid DESC LIMIT 100",
                    (module, module),
                )
            ]

    def get_record(self, record_id: str) -> dict:
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM analysis_records WHERE id=?", (record_id,)
            ).fetchone()
        if row is None:
            raise KeyError("分析记录不存在")
        return json.loads(row[0])
