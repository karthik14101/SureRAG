"""An oversized upload must be rejected without being buffered first.

`UploadFile` spools to disk while the request arrives, but `.read()` with no
argument materialises the whole file as one bytes object -- and the size check
used to live in staging, which runs afterwards. A 2 GB file was therefore fully
resident in memory before anything rejected it.

Async work is wrapped in `asyncio.run` inside sync test functions so the suite
needs neither pytest-asyncio nor any other plugin.
"""
from __future__ import annotations

import asyncio
import io
import tracemalloc

from starlette.datastructures import UploadFile

from app.api.routes.documents import MAX_REQUEST_BYTES, READ_CHUNK_BYTES, _read_capped
from app.config import settings

MB = 1024 * 1024


class LazyBytes:
    """A file object that produces `size` bytes without ever holding them.

    Building a real 64 MB buffer would allocate 64 MB in the test itself, so the
    memory measurement below would be reading its own fixture rather than the
    code under test. This was a genuine false failure before it was fixed.
    """

    def __init__(self, size):
        self.size, self.pos = size, 0

    def read(self, n=-1):
        left = self.size - self.pos
        if left <= 0:
            return b""
        take = left if n is None or n < 0 else min(n, left)
        self.pos += take
        return b"x" * take

    def seek(self, *_a, **_k):
        self.pos = 0
        return 0

    def close(self):
        pass


def upload(size, name="big.pdf", lazy=False):
    source = LazyBytes(size) if lazy else io.BytesIO(b"x" * size)
    return UploadFile(file=source, filename=name)


def test_a_file_inside_the_limit_reads_normally():
    data = asyncio.run(_read_capped(upload(3 * MB), 10 * MB))
    assert data is not None and len(data) == 3 * MB


def test_a_file_over_the_limit_is_rejected_not_truncated():
    over = asyncio.run(_read_capped(upload(12 * MB), 10 * MB))
    # Truncating would be worse than rejecting: a half-read PDF still passes the
    # magic-byte check and would be ingested as a silently corrupt document.
    assert over is None, type(over)


def test_the_boundary_is_inclusive():
    exact = asyncio.run(_read_capped(upload(4 * MB), 4 * MB))
    assert exact is not None and len(exact) == 4 * MB
    assert asyncio.run(_read_capped(upload(4 * MB + 1), 4 * MB)) is None


def test_rejecting_does_not_hold_the_whole_file_in_memory():
    """The point of the whole change."""
    lazy = upload(64 * MB, "huge.pdf", lazy=True)

    tracemalloc.start()
    rejected = asyncio.run(_read_capped(lazy, 8 * MB))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert rejected is None
    ceiling = 8 * MB + 4 * READ_CHUNK_BYTES
    assert peak < ceiling, (
        "peak {:.1f} MB against a {:.0f} MB ceiling -- the old code held all "
        "64 MB before the size check ran".format(peak / MB, ceiling / MB)
    )


def test_empty_and_tiny_files_still_behave():
    assert asyncio.run(_read_capped(upload(0), 10 * MB)) == b""
    assert asyncio.run(_read_capped(upload(10), 10 * MB)) == b"x" * 10


def test_the_request_budget_always_admits_one_maximum_size_file():
    assert MAX_REQUEST_BYTES >= settings.max_upload_bytes, (
        MAX_REQUEST_BYTES,
        settings.max_upload_bytes,
    )


def test_a_batch_stops_once_the_request_budget_is_spent():
    """The route's loop: capping each file still allowed fifty of them to add
    up to something that kills the process, so the budget shrinks as files are
    accepted and later ones are refused."""

    async def run_batch():
        remaining, accepted, refused = 10 * MB, [], []
        for index, size in enumerate([4 * MB, 4 * MB, 4 * MB]):
            content = await _read_capped(upload(size), min(6 * MB, remaining))
            if content is None:
                refused.append(index)
                continue
            remaining -= len(content)
            accepted.append(index)
        return accepted, refused

    accepted, refused = asyncio.run(run_batch())
    assert accepted == [0, 1] and refused == [2], (accepted, refused)
