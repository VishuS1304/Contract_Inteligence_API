"""
pytest fixtures for Contract Intelligence API tests.

- Create a temporary STORE_DIR for tests (session-scoped).
- Patch environment using a session-scoped pytest.MonkeyPatch instance.
- Delay importing the FastAPI app until after STORE_DIR is set (avoids modules
  reading STORE_DIR at import-time and writing to the wrong location).
- Provide a TestClient wrapper that accepts and drops `stream=True` kwarg.
"""

import os
import time
import json
import shutil
import logging
from pathlib import Path
import pytest

from fastapi.testclient import TestClient

LOGGER = logging.getLogger("contract-intel.tests")

# NOTE: Do NOT import app.main at top-level; import inside client fixture after env patching.


@pytest.fixture(scope="session")
def tmp_store_dir(tmp_path_factory):
    """
    Create a temporary store directory (session-scoped).
    Tests expect STORE_DIR env var to point to a path with pdfs/ and texts/.
    """
    base = tmp_path_factory.mktemp("store")
    pdfs = base / "pdfs"
    texts = base / "texts"
    pdfs.mkdir(exist_ok=True)
    texts.mkdir(exist_ok=True)
    # create an empty index.json
    (base / "index.json").write_text(json.dumps({}))
    yield base
    # cleanup after session
    try:
        shutil.rmtree(base)
    except Exception:
        LOGGER.warning("Failed to cleanup tmp store dir %s", base)


@pytest.fixture(scope="session")
def patch_store_env(tmp_store_dir):
    """
    Patch environment variables used by the application so tests save files into tmp_store_dir.

    Use pytest.MonkeyPatch() manually so this fixture can be session-scoped without depending
    on the function-scoped monkeypatch fixture (avoids ScopeMismatch).
    """
    mp = pytest.MonkeyPatch()
    try:
        mp.setenv("STORE_DIR", str(tmp_store_dir))
        mp.setenv("DATA_DIR", "data")
        # ensure directories exist, in case tests expect them immediately
        Path(os.environ["STORE_DIR"]).mkdir(parents=True, exist_ok=True)
        (Path(os.environ["STORE_DIR"]) / "pdfs").mkdir(exist_ok=True)
        (Path(os.environ["STORE_DIR"]) / "texts").mkdir(exist_ok=True)
        yield
    finally:
        # restore environment
        mp.undo()


@pytest.fixture(scope="session")
def client(patch_store_env):
    """
    Create the TestClient AFTER patch_store_env has run, so app modules that read STORE_DIR at import
    time will pick up the patched path. Wrap TestClient.get to accept `stream=True`.
    """
    # import app only after patch_store_env has set STORE_DIR
    import importlib
    app_mod = importlib.import_module("app.main")
    app = getattr(app_mod, "app")

    tc = TestClient(app)

    # wrap .get to accept a `stream` kwarg (tests pass it)
    orig_get = tc.get

    def get_with_stream(path, *args, **kwargs):
        kwargs.pop("stream", None)
        return orig_get(path, *args, **kwargs)

    tc.get = get_with_stream

    # wrap .post as well (harmless)
    orig_post = tc.post

    def post_with_stream(path, *args, **kwargs):
        kwargs.pop("stream", None)
        return orig_post(path, *args, **kwargs)

    tc.post = post_with_stream

    yield tc

    try:
        tc.close()
    except Exception:
        pass
