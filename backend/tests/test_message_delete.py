"""Deleting a turn from a chat thread.

A question takes the answer it produced with it; the rest of the thread is left
alone. The thread's name follows its first question, which is the part that was
broken: a thread is named once and `maybe_title_session` refuses to rename one
that already has a name, so deleting the question it was named after left the
sidebar advertising something no longer there.
"""
from __future__ import annotations

import pathlib
import tempfile
from datetime import datetime, timedelta

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import models
from app.db.base import Base
from app.services import chat_service

_engine = create_engine("sqlite:///{}".format(pathlib.Path(tempfile.mkdtemp()) / "t.db"))
_Session = sessionmaker(bind=_engine)


def fresh_thread():
    """A three-turn thread: Q1 A1 Q2 A2 Q3 A3, a citation on each answer.

    Timestamps are set explicitly and increasing. Inserting six rows in a tight
    loop puts them all in the same microsecond, and ordering then falls back to
    the primary key -- which is a random id, not a sequence. Real turns are
    seconds apart because an answer takes seconds to generate.
    """
    Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)
    db = _Session()

    db.add(models.ChatSession(id="s1", user_id="u1", kb_id="kb1", title="t", message_count=0))
    db.flush()

    ids = []
    clock = datetime(2026, 1, 1, 12, 0, 0)
    for turn in range(1, 4):
        question = models.Message(
            session_id="s1", role="user", content=f"question {turn}",
            created_at=clock + timedelta(seconds=turn * 60),
        )
        db.add(question)
        db.flush()
        answer = models.Message(
            session_id="s1", role="assistant", content=f"answer {turn}",
            created_at=clock + timedelta(seconds=turn * 60 + 20),
        )
        db.add(answer)
        db.flush()
        db.add(models.MessageCitation(
            message_id=answer.id, marker_index=1, chunk_id=f"c{turn}",
            doc_id="d", filename="f.json", snippet="s", score=0.5,
        ))
        ids += [question.id, answer.id]

    db.get(models.ChatSession, "s1").message_count = 6
    db.commit()
    return db, ids


def contents(db):
    rows = db.execute(
        select(models.Message)
        .where(models.Message.session_id == "s1")
        .order_by(models.Message.created_at.asc(), models.Message.id.asc())
    ).scalars().all()
    return [row.content for row in rows]


def count(db):
    return db.get(models.ChatSession, "s1").message_count


def title(db):
    return db.get(models.ChatSession, "s1").title


def test_deleting_a_question_takes_its_answer_with_it():
    db, ids = fresh_thread()
    removed = chat_service.delete_message(db, "s1", ids[2])  # question 2

    assert removed == 2, removed
    assert contents(db) == ["question 1", "answer 1", "question 3", "answer 3"], contents(db)
    assert count(db) == 4, count(db)


def test_deleting_the_last_question_needs_no_answer_after_it():
    db, ids = fresh_thread()

    assert chat_service.delete_message(db, "s1", ids[4]) == 2
    assert contents(db) == ["question 1", "answer 1", "question 2", "answer 2"], contents(db)


def test_deleting_several_turns_in_a_row_stays_consistent():
    db, ids = fresh_thread()
    chat_service.delete_message(db, "s1", ids[0])
    chat_service.delete_message(db, "s1", ids[2])

    assert contents(db) == ["question 3", "answer 3"], contents(db)
    assert count(db) == 2, count(db)


def test_deleting_an_answer_on_its_own_leaves_the_question():
    db, ids = fresh_thread()

    assert chat_service.delete_message(db, "s1", ids[1]) == 1
    assert contents(db)[:2] == ["question 1", "question 2"], contents(db)


def test_citations_go_with_their_message():
    db, ids = fresh_thread()
    before = db.query(models.MessageCitation).count()
    chat_service.delete_message(db, "s1", ids[0])

    assert before == 3, before
    # The deleted answer's citation goes with it rather than being orphaned.
    assert db.query(models.MessageCitation).count() == 2


def test_the_rolling_summary_is_dropped_not_left_stale():
    db, ids = fresh_thread()
    db.get(models.ChatSession, "s1").history_summary = "Earlier the user asked about question 2."
    db.commit()

    chat_service.delete_message(db, "s1", ids[2])

    # Otherwise the next answer could cite a question the user just removed.
    assert db.get(models.ChatSession, "s1").history_summary is None


def test_bad_input_changes_nothing():
    db, ids = fresh_thread()

    assert chat_service.delete_message(db, "s1", "nope") == 0
    assert len(contents(db)) == 6

    # A message id from another thread must not be reachable through this one.
    assert chat_service.delete_message(db, "other-session", ids[0]) == 0
    assert len(contents(db)) == 6


def test_the_thread_name_follows_its_first_question():
    db, ids = fresh_thread()
    db.get(models.ChatSession, "s1").title = "Who Is CEO Of Google"
    db.commit()

    chat_service.delete_message(db, "s1", ids[0])
    assert title(db) == "question 2", title(db)

    chat_service.delete_message(db, "s1", ids[2])
    assert title(db) == "question 3", title(db)

    chat_service.delete_message(db, "s1", ids[4])
    # An emptied thread goes back to the default, which is the one state
    # maybe_title_session will act on, so the next question names it properly.
    assert title(db) == chat_service.DEFAULT_TITLE, title(db)


def test_deleting_a_later_turn_leaves_the_name_alone():
    db, ids = fresh_thread()
    db.get(models.ChatSession, "s1").title = "Question One"
    db.commit()

    chat_service.delete_message(db, "s1", ids[2])
    assert title(db) == "Question One", title(db)


def test_message_count_never_goes_negative():
    db, ids = fresh_thread()
    db.get(models.ChatSession, "s1").message_count = 1
    db.commit()

    chat_service.delete_message(db, "s1", ids[0])
    assert count(db) == 0, count(db)
