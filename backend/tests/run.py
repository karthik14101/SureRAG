"""Run the test suite without pytest.

    python tests/run.py                  # everything
    python tests/run.py stop_rules       # only files matching a name

The suite is written as ordinary pytest files -- `test_*` functions and plain
asserts -- so `pytest` works too once it is installed. This runner exists so
that it does not have to be: a test suite you cannot run until you install
something is a test suite that gets skipped on the day it matters.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tests._support import reset_providers, restore_settings, snapshot_settings

HERE = pathlib.Path(__file__).resolve().parent


def discover(pattern: str | None):
    for path in sorted(HERE.glob("test_*.py")):
        if pattern and pattern not in path.stem:
            continue
        yield path


def run_module(path: pathlib.Path) -> tuple[int, list[str]]:
    module = importlib.import_module(f"tests.{path.stem}")
    tests = [
        (name, fn)
        for name, fn in vars(module).items()
        if name.startswith("test_") and callable(fn)
    ]
    # Definition order, which is the order the sections were meant to be read.
    tests.sort(key=lambda item: inspect.getsourcelines(item[1])[1])

    passed, failures = 0, []
    for name, fn in tests:
        saved = snapshot_settings()
        reset_providers()
        try:
            result = fn()
            if inspect.iscoroutine(result):
                asyncio.run(result)
            passed += 1
            print(f"  PASS  {name}")
        except Exception:
            failures.append(f"{path.stem}::{name}\n{traceback.format_exc()}")
            print(f"  FAIL  {name}")
        finally:
            restore_settings(saved)
            reset_providers()
    return passed, failures


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else None
    total, all_failures = 0, []

    for path in discover(pattern):
        print(f"\n{path.stem}")
        passed, failures = run_module(path)
        total += passed
        all_failures.extend(failures)

    print("\n" + "=" * 64)
    if all_failures:
        for failure in all_failures:
            print("\n" + failure)
        print(f"  {total} passed, {len(all_failures)} FAILED")
    else:
        print(f"  {total} passed")
    print("=" * 64)
    return 1 if all_failures else 0


if __name__ == "__main__":
    sys.exit(main())
