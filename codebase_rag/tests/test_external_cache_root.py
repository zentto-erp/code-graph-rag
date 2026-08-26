from __future__ import annotations

import stat
from pathlib import Path

import pytest

from codebase_rag import constants as cs
from codebase_rag.cli import _delete_hash_cache
from codebase_rag.config import settings
from codebase_rag.tests.conftest import create_and_run_updater
from codebase_rag.utils.path_utils import resolve_index_state_paths


def _state_filenames() -> tuple[str, str, str]:
    return (
        cs.HASH_CACHE_FILENAME,
        cs.DIR_MTIMES_FILENAME,
        cs.PARSER_FINGERPRINT_FILENAME,
    )


def test_default_state_paths_remain_inside_repo(tmp_path: Path) -> None:
    state = resolve_index_state_paths(tmp_path, "project")

    assert state.directory == tmp_path
    assert state.hash_cache == tmp_path / cs.HASH_CACHE_FILENAME
    assert state.dir_mtimes == tmp_path / cs.DIR_MTIMES_FILENAME
    assert state.parser_fingerprint == tmp_path / cs.PARSER_FINGERPRINT_FILENAME


def test_external_state_is_deterministic_and_isolated(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    repo_a = tmp_path / "a" / "repo"
    repo_b = tmp_path / "b" / "repo"
    repo_a.mkdir(parents=True)
    repo_b.mkdir(parents=True)

    first = resolve_index_state_paths(repo_a, "same-project", cache_root)
    repeated = resolve_index_state_paths(repo_a, "same-project", cache_root)
    other_repo = resolve_index_state_paths(repo_b, "same-project", cache_root)
    other_project = resolve_index_state_paths(repo_a, "other-project", cache_root)

    assert first == repeated
    assert len({first.directory, other_repo.directory, other_project.directory}) == 3
    assert first.directory.parent == cache_root.resolve()


def test_project_name_cannot_traverse_external_cache_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    repo = tmp_path / "repo"
    repo.mkdir()

    state = resolve_index_state_paths(
        repo, "../../outside/../escape\\project", cache_root
    )

    assert state.directory.parent == cache_root.resolve()
    assert state.directory.is_relative_to(cache_root.resolve())
    assert ".." not in state.directory.name


def test_indexing_read_only_source_writes_all_state_externally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_ingestor,
) -> None:
    repo = tmp_path / "read-only-source"
    repo.mkdir()
    (repo / "module.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    cache_root = tmp_path / "external-cache"
    project_name = repo.name
    monkeypatch.setattr(settings, "CACHE_ROOT", cache_root)

    original_mode = stat.S_IMODE(repo.stat().st_mode)
    repo.chmod(stat.S_IREAD | stat.S_IEXEC)
    try:
        create_and_run_updater(repo, mock_ingestor, skip_if_missing=None)
    finally:
        repo.chmod(original_mode)

    state = resolve_index_state_paths(repo, project_name, cache_root)
    assert all((state.directory / name).is_file() for name in _state_filenames())
    assert all(not (repo / name).exists() for name in _state_filenames())


def test_clean_removes_external_state_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    cache_root = tmp_path / "cache"
    project_name = "project"
    monkeypatch.setattr(settings, "CACHE_ROOT", cache_root)
    state = resolve_index_state_paths(repo, project_name, cache_root)
    state.directory.mkdir(parents=True)
    for name in _state_filenames():
        (state.directory / name).write_text(cs.JSON_EMPTY_OBJECT, encoding="utf-8")
        (repo / name).write_text("source sentinel", encoding="utf-8")

    _delete_hash_cache(repo, project_name)

    assert all(not (state.directory / name).exists() for name in _state_filenames())
    assert all((repo / name).is_file() for name in _state_filenames())
