"""The corpus-shape hint the verifier is given.

The verifier is told which files a knowledge base holds but nothing about what
is inside them, and that gap cost real turns: asked how Article 368 relates to
the Seventh Schedule it demanded "the text of the Seventh Schedule itself" on
every pass -- correctly, because that corpus is article records and the
schedules were never ingested, but with no way to tell that from a retrieval
failure. So it kept searching for something that does not exist.
"""
from __future__ import annotations

import asyncio

from app.vectorstore import search

ARTICLE = "[{i}].article = {i} [{i}].title = Some title [{i}].description = Body text here."
PROSE = "The quick brown fox jumped over the lazy dog, and then kept going for a while."


class Point:
    def __init__(self, text, **payload):
        self.payload = {"text": text, **payload}


def manual(n, modality="text"):
    """Passages as a PDF owner's manual produces them: paged, no field markers."""
    return [
        Point(PROSE, filename="curvv-ev-owners-manual.pdf", page_no=1 + i, modality=modality)
        for i in range(n)
    ]


class FakeClient:
    """Stands in for Qdrant, counting calls so caching can be observed."""

    def __init__(self, points, fail=False):
        self.points, self.fail, self.calls = points, fail, 0

    async def scroll(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("All connection attempts failed")
        return self.points[: kwargs.get("limit", 200)], None


def install(points, fail=False):
    client = FakeClient(points, fail)
    search.get_client = lambda: client
    search._corpus_shapes.clear()
    return client


def articles(n):
    return [Point(ARTICLE.format(i=i)) for i in range(n)]


def test_a_uniform_structured_corpus_is_described_exactly():
    install(articles(200))
    shape = asyncio.run(search.describe_corpus_shape("kb", "rev1"))

    assert "article, title, description" in shape, shape
    assert "200" in shape, shape
    # "Every one" is only claimed when it is literally true.
    assert shape.startswith("Every one of the 200"), shape


def test_a_mixed_corpus_is_reported_as_mixed_not_rounded_up():
    """A handful of passages usually carry no field markers at all -- a preamble,
    the tail of a record split across chunks. Claiming "every one" over them
    would be a plain untruth in the one place the verifier cannot check."""
    install(articles(190) + [Point(PROSE)] * 10)
    shape = asyncio.run(search.describe_corpus_shape("kb", "rev2"))

    assert not shape.startswith("Every one"), shape
    assert shape.startswith("190 of the 200"), shape
    assert "carry no field markers" in shape, shape
    assert "article, title, description" in shape, shape


def test_a_prose_corpus_is_described_from_what_its_payloads_carry():
    """Prose used to get no sentence at all, which is how a 364-page car manual
    reached the verifier as "(not described)". The probe then fired on one JSON
    corpus and nothing else, because hardly any corpus is JSON."""
    install(manual(200))
    shape = asyncio.run(search.describe_corpus_shape("kb", "rev3"))

    assert "1 file spanning pages 1 to 200" in shape, shape
    assert "all are running text" in shape, shape


def test_the_mix_of_passage_kinds_is_reported():
    """The useful half on a manual. Tables arrive flattened and figures arrive as
    their captions, so a verifier told this can reason about why a pictogram is
    nowhere described instead of concluding the topic is absent."""
    install(manual(120, "text") + manual(60, "table") + manual(20, "caption"))
    shape = asyncio.run(search.describe_corpus_shape("kb", "rev3b"))

    assert "120 are running text" in shape, shape
    assert "60 are tables rendered as text" in shape, shape
    assert "and 20 are figure captions" in shape, shape


def test_payloads_that_carry_nothing_identifying_stay_silent():
    """Without a filename or a page number there is nothing to say that is not
    guesswork, and a guess is worse than silence in the one place the verifier
    cannot check."""
    install([Point(PROSE)] * 200)
    assert asyncio.run(search.describe_corpus_shape("kb", "rev3c")) == ""


def test_too_much_variety_to_summarise_means_no_claim():
    mixed = [Point(f"[{i}].alpha = x [{i}].beta = y") for i in range(100)]
    mixed += [Point(f"[{i}].gamma = x [{i}].delta = y") for i in range(100)]
    install(mixed)

    assert asyncio.run(search.describe_corpus_shape("kb", "rev4")) == ""


def test_a_dead_vector_store_degrades_to_silence_never_to_an_error():
    client = install([], fail=True)
    # Verified for real: Qdrant happened to be stopped while this was written.
    assert asyncio.run(search.describe_corpus_shape("kb", "rev5")) == ""
    assert client.calls == 1


def test_measured_once_per_knowledge_base_not_once_per_turn():
    client = install(articles(200))
    for _ in range(5):
        asyncio.run(search.describe_corpus_shape("kb", "rev6"))
    assert client.calls == 1, client.calls

    # The profile carries the passage count, so re-ingestion changes the key.
    asyncio.run(search.describe_corpus_shape("kb", "rev6-after-reingest"))
    assert client.calls == 2, client.calls

    asyncio.run(search.describe_corpus_shape("other-kb", "rev6"))
    assert client.calls == 3, client.calls


def test_the_sentence_states_shape_and_does_not_editorialise():
    install(articles(200))
    shape = asyncio.run(search.describe_corpus_shape("kb", "rev7"))

    # Nothing here mentions schedules: a corpus of article records plainly has
    # none, and that is the inference the sentence makes available rather than
    # draws itself.
    assert "schedule" not in shape.lower(), shape
    assert "missing" not in shape.lower(), shape
    assert shape.count(". ") <= 1, shape
