"""
SURE-GraphRAG backend entrypoint.

Run from the `backend/` directory with your virtualenv active:

    python run.py

Equivalent to `uvicorn app.main:app --reload`, but reads host/port from .env and
sets the HuggingFace cache location before torch/transformers are imported.
"""
from __future__ import annotations

import os
import pathlib
import sys

# The .env lives one level up (project root). Load it *before* importing the app
# so HF_HOME is in place before sentence-transformers is first touched.
ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ENV_FILE)

_hf_home = os.getenv("HF_HOME", "./.hf_cache")
_hf_path = (ROOT / _hf_home).resolve() if not os.path.isabs(_hf_home) else pathlib.Path(_hf_home)
_hf_path.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_hf_path)
# Silences the noisy symlink warning on Windows, where symlinks need admin rights.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import uvicorn  # noqa: E402


def main() -> int:
    host = os.getenv("BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("BACKEND_PORT", "8000"))
    reload_enabled = os.getenv("BACKEND_RELOAD", "true").lower() in {"1", "true", "yes"}
    log_level = os.getenv("LOG_LEVEL", "INFO").lower()

    # 0.0.0.0 and :: are bind-all wildcards, not addresses a browser can open.
    # Always advertise a URL that actually resolves.
    browsable = "127.0.0.1" if host in ("0.0.0.0", "::", "*", "") else host

    print("=" * 70)
    print("  SURE-GraphRAG backend")
    print(f"  http://{browsable}:{port}        docs: http://{browsable}:{port}/docs")
    print(f"  provider: {os.getenv('LLM_PROVIDER', 'gemini')}   env: {ENV_FILE}")
    if host != browsable:
        print(f"  (listening on {host}:{port} -- reachable from other devices)")
    print("=" * 70)

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload_enabled,
        log_level=log_level,
        # Reloading on the model cache or the SQLite journal causes restart loops.
        reload_excludes=["*.db", "*.db-wal", "*.db-shm", ".hf_cache/*", "../data/*"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
