"""Out-of-scope detection, and measuring how grounded an answer is.

Two numbers that answer different questions. Sufficiency judges the evidence
pool before a word is written and is reliably pessimistic on questions whose
answer has to be assembled from several passages. Grounding judges the answer
that was actually written. They are allowed to disagree; the disagreement is
the point.
"""
from __future__ import annotations

import pathlib

from app.agent import orchestrator
from app.agent.nodes import synthesize, verifier
from app.agent.state import AgentState, Citation
from app.vectorstore.search import RetrievedChunk


def cite(i):
    return Citation(marker_index=i, chunk=RetrievedChunk(
        chunk_id=f"c{i}", doc_id="d", kb_id="kb", filename="f.json", text="t", score=0.5))


# What the shape probe reports for the corpus these turns were measured on.
# Concluding that a corpus lacks a kind of material requires having established
# what it does hold, so the guard below asks for this before drawing it.
SHAPE = (
    "Every one of the 200 passages sampled from this corpus is a structured "
    "record with exactly these fields: article, title, description."
)


def state_after_pass(score, previous, iterations=1, shape=SHAPE, in_scope=None):
    state = AgentState(question="q", user_id="u", kb_id="kb", session_id="s")
    state.sufficiency_score, state.previous_score, state.iterations = score, previous, iterations
    state.last_expansion_gain, state.last_judged_turnover = 18, 0.67
    state.corpus_shape, state.verifier_in_scope = shape, in_scope
    return state


# ---------------------------------------------------------------------------
# in_scope
# ---------------------------------------------------------------------------
def test_an_omitted_in_scope_field_is_not_read_as_in_scope():
    """`payload.get("in_scope") is False` was False for a missing key, which is
    indistinguishable from the model saying the question IS in scope -- so the
    out-of-scope path never once fired on live traffic."""
    assert verifier._tri_state(False) is False
    assert verifier._tri_state(True) is True
    assert verifier._tri_state(None) is None
    assert verifier._tri_state({}.get("in_scope")) is None
    assert verifier._tri_state("false") is False
    assert verifier._tri_state("No") is False
    assert verifier._tri_state("true") is True
    assert verifier._tri_state("maybe") is None

    # The bug itself: the old expression could not tell the two cases apart.
    assert ({}.get("in_scope") is False) == ({"in_scope": True}.get("in_scope") is False)


def test_a_floor_score_after_a_real_expansion_pass_means_out_of_scope():
    """The MRP turn: 20% then 10% after targeted queries, a hypothetical-answer
    probe and a graph widening all ran. The verifier never set in_scope."""
    state = state_after_pass(score=0.10, previous=0.20)

    assert orchestrator._should_stop(state)
    assert state.out_of_scope
    assert "does not appear to hold" in state.stop_reason, state.stop_reason


def test_an_explicit_in_scope_verdict_is_not_overruled_by_the_score():
    """The Curvv EV turtle-lamp turn. The manual answers the question -- "Limp
    Home Mode | Amber", and a table giving that continuous chime at 5% charge --
    but across 364 pages it never writes the word "turtle", so the verifier
    scored 20% twice while its own covered list named limp home mode. It also
    said, both times, that the question was findable here. The loop overruled it
    on the strength of a number and told the reader the manual had no such
    material."""
    state = state_after_pass(score=0.20, previous=0.20, in_scope=True)

    assert orchestrator._should_stop(state)          # the score still did not rise
    assert not state.out_of_scope                    # but absence was never established
    assert "did not improve" in state.stop_reason, state.stop_reason


def test_nothing_is_declared_absent_from_a_corpus_that_was_never_described():
    """The same turn, second guard. The verifier was shown "(not described)" --
    the shape probe only ever matched JSON records -- and the loop pronounced on
    what the corpus did not contain anyway."""
    state = state_after_pass(score=0.10, previous=0.20, shape="")

    assert orchestrator._should_stop(state)
    assert not state.out_of_scope
    assert "did not improve" in state.stop_reason, state.stop_reason


def test_a_merely_thin_answer_is_not_called_out_of_scope():
    """The Article 368 turn: 50% then 50%. Real evidence, pessimistic verifier."""
    state = state_after_pass(score=0.50, previous=0.50)

    assert orchestrator._should_stop(state)
    assert not state.out_of_scope
    assert "did not improve" in state.stop_reason, state.stop_reason


def test_a_low_score_before_any_expansion_still_gets_its_pass():
    state = AgentState(question="q", user_id="u", kb_id="kb", session_id="s")
    state.sufficiency_score, state.previous_score = 0.10, None

    assert not orchestrator._should_stop(state)
    assert not state.out_of_scope


def test_a_rising_score_is_never_cut_off_as_out_of_scope():
    state = state_after_pass(score=0.24, previous=0.05)

    assert not orchestrator._should_stop(state)
    assert not state.out_of_scope, state.stop_reason


# ---------------------------------------------------------------------------
# grounding
# ---------------------------------------------------------------------------
CITED = (
    "The structural link is that any amendment which seeks to change a List in "
    "the Seventh Schedule must be ratified by half the States. [1]\n"
    "Article 246 defines the Union, State and Concurrent Lists as the bases of "
    "legislative power for Parliament and the State Legislatures. [2]\n"
    "Taken together, altering those Lists requires both the special majority and "
    "the ratification prescribed by Article 368(2). [1][2]\n"
)


def test_grounding_measures_the_answer_not_the_verifiers_appetite():
    assert synthesize.measure_grounding(CITED, [cite(1), cite(2)]) == 1.0

    half = (
        "Article 359 permits the President to suspend enforcement of the rights "
        "mentioned in the order for the duration of the Emergency. [1]\n"
        "It is widely believed that this power has been used sparingly in practice "
        "and that the courts have pushed back on it repeatedly.\n"
    )
    score = synthesize.measure_grounding(half, [cite(1)])
    assert 0.4 < score < 0.6, score


def test_grounding_degrades_sensibly():
    assert synthesize.measure_grounding(
        "A long uncited assertion about the Constitution and its many provisions.", []
    ) == 0.0
    assert synthesize.measure_grounding("   ", [cite(1)]) is None
    assert synthesize.measure_grounding("# Heading\nYes.\nNo.", [cite(1)]) is None
    assert synthesize.measure_grounding(
        "## A heading long enough to pass the length test on its own merits\n"
        "A substantive sentence that does carry a marker for its claim. [1]\n",
        [cite(1)],
    ) == 1.0


def test_quoted_statute_full_of_ellipses_is_not_a_sentence_boundary():
    """Treating each dot of "... any of the Lists ..." as a sentence end scored
    a complete, fully cited answer at 57%."""
    q2 = (
        'The structural link is that any amendment Bill which "seeks to make any '
        'change in ... any of the Lists in the Seventh Schedule" must follow a '
        "special, federally ratified procedure under the proviso to Article "
        "368(2).[1]\n"
        "Article 246(1)-(3) defines the Union List, State List and Concurrent List "
        "in the Seventh Schedule as the bases for the respective legislative powers "
        "of Parliament and the State Legislatures.[2]\n"
        "Taken together, any constitutional amendment that alters those Lists must "
        "undergo both the special parliamentary majority and mandatory State "
        "Legislature ratification prescribed by Article 368(2).[1][2]\n"
        "The sources do not further explain this linkage in doctrinal or "
        "theoretical terms beyond the bare constitutional text."
    )
    # The verifier called this answer 50%; grounding sees it whole.
    assert synthesize.measure_grounding(q2, [cite(1), cite(2)]) == 1.0

    assert len(synthesize._split_sentences(
        'A quote "change in ... any of the Lists" ends here.[1]')) == 1
    assert len(synthesize._split_sentences(
        "First one ends here.[1] Second one is capitalised and ends.[2]")) == 2
    # A stop before a number must not split, or "Art. 45" becomes two claims.
    assert len(synthesize._split_sentences(
        "The duty appears in Art. 45 of the Constitution and nowhere else.[1]")) == 1


def test_a_sentence_about_the_evidence_is_not_an_uncited_claim():
    """THIN_EVIDENCE_NOTE asks for exactly this sentence, and the old measure
    then docked the answer for writing it."""
    for meta in [
        "What is missing in these sources is any detailed description of the "
        "specific subjects in each List or illustrative examples.",
        "The sources do not further explain this linkage in doctrinal terms.",
        "This knowledge base does not show any constitutional article about MRP.",
        "None of these extracts deals with consumer protection or pricing.",
        "The available sources state the content but do not specify the dates.",
        "The provided material does not reproduce Article 51A or its clauses.",
    ]:
        assert synthesize._is_about_the_evidence(meta), meta

    for claim in [
        "Article 359 allows the President to suspend the right to move any court "
        "for enforcement of specified Fundamental Rights.",
        # Mentions "context" and "no", and must still count as a real claim.
        "In the context of Article 21, no person shall be deprived of life or "
        "personal liberty except according to procedure established by law.",
        "Article 20 and Article 21 are protected and cannot be suspended by any "
        "order made under Article 359 during an Emergency.",
    ]:
        assert not synthesize._is_about_the_evidence(claim), claim


def test_an_answer_that_is_nothing_but_absence_has_nothing_to_ground():
    only_meta = (
        "This knowledge base does not hold any provision about maximum retail "
        "price or overcharging by retailers at railway stations."
    )
    # Reporting 100% there would be a boast about saying nothing.
    assert synthesize.measure_grounding(only_meta, [cite(1)]) is None


# ---------------------------------------------------------------------------
# lists
# ---------------------------------------------------------------------------
LISTED = (
    "The structural link is that any amendment changing a List in the Seventh "
    "Schedule needs ratification by half the States.[1]\n"
    "\n"
    "The Lists structurally define the subject-matter of laws made by Parliament "
    "and by the Legislatures of States, since Article 246(1)-(3) grants:\n"
    "\n"
    "- Parliament exclusive power to make laws on matters in List I (Union List),\n"
    "- concurrent power to Parliament and State Legislatures over List III, and\n"
    "- exclusive power to State Legislatures over matters in List II.[2]\n"
)


def test_a_list_and_its_lead_in_are_one_evidential_unit():
    """Live answer shape: a colon lead-in, bullets drawn from one source, the
    citation on the final bullet. Counting each bullet separately read that as
    three unsourced assertions and rated a correct answer at 50%."""
    units = synthesize._claim_units(LISTED)

    assert len(units) == 2, units
    assert units[1][0] is True, units[1]
    # A blank line between a lead-in and its bullets must not split them.
    assert "Article 246" in units[1][1] and "List II" in units[1][1]
    assert synthesize.measure_grounding(LISTED, [cite(1), cite(2)]) == 1.0


def test_an_entirely_uncited_list_is_still_penalised():
    """Grouping must not become a free pass."""
    unsourced = (
        "Here is the position on legislative competence in the Constitution:\n"
        "\n"
        "- Parliament has exclusive power over matters in the Union List there,\n"
        "- State Legislatures have exclusive power over the State List matters,\n"
        "- both may legislate on the Concurrent List subject to Union primacy.\n"
        "\n"
        "Article 246 sets out this threefold distribution of legislative power.[1]\n"
    )
    assert synthesize.measure_grounding(unsourced, [cite(1)]) == 0.5


def test_numbered_items_group_like_bullets_and_a_heading_breaks_a_list():
    assert len(synthesize._claim_units(
        "Intro line that runs on:\n1. first item\n2. second item")) == 1
    assert len(synthesize._claim_units(
        "Lead in here:\n- one item\n# Heading\n- another item")) == 2


# ---------------------------------------------------------------------------
# tiering
# ---------------------------------------------------------------------------
def test_every_answer_is_written_by_the_main_model():
    """The fast tier was tried for out-of-scope answers and reverted: naming
    what a corpus holds instead of the answer is a relevance judgement, and the
    cheap model offered "Grants in lieu of export duty on jute and jute
    products" in reply to a question about MRP."""
    module_text = pathlib.Path(synthesize.__file__.replace(".pyc", ".py")).read_text(
        encoding="utf-8"
    )

    assert "get_fast_llm" not in module_text
    # The reasoning is recorded where someone would otherwise re-add it.
    assert "jute" in module_text


def test_the_two_numbers_are_independent_and_both_survive_the_turn():
    state = AgentState(question="q", user_id="u", kb_id="kb", session_id="s")
    state.sufficiency_score = 0.5
    state.grounding = synthesize.measure_grounding(CITED, [cite(1), cite(2)])

    assert state.sufficiency_score == 0.5
    assert state.grounding == 1.0
    assert state.grounding is not None and state.sufficiency_score is not None
