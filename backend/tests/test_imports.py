"""Every module imports.

Cheap and unglamorous, and it has caught real breakage: a function moved
between modules, an import left behind after a refactor, a circular import
introduced by adding one convenience import at the top of a file. None of the
behavioural tests below would notice, because they only import what they use.
"""
from __future__ import annotations

import importlib
import pathlib

APP_ROOT = pathlib.Path(__file__).resolve().parent.parent / "app"

# Optional heavy dependencies. A missing one is a setup choice, not a bug: the
# providers are selected by configuration and only the chosen one is installed.
OPTIONAL = ("torch", "transformers", "bitsandbytes", "accelerate")


def module_names():
    for path in sorted(APP_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(APP_ROOT.parent).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        yield ".".join(parts)


def test_every_app_module_imports():
    failures = []
    skipped = []

    for name in module_names():
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] in OPTIONAL:
                skipped.append(f"{name} (needs {exc.name})")
                continue
            failures.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - any import-time error counts
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    assert not failures, "modules failed to import:\n  " + "\n  ".join(failures)
    # A sanity floor, so an empty walk cannot pass silently.
    assert len(list(module_names())) > 50, len(list(module_names()))
