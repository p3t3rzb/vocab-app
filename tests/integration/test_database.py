"""Engine lifecycle, migrations, repositories and the raw sqlite peek helpers."""

import sqlite3

import pytest
from sqlalchemy import select

from src.database import (
    Direction,
    LanguagePairRepository,
    RepetitionRepository,
    Word,
    WordRepository,
    count_words,
    get_session,
    init_db,
    read_language_pair,
)
from src.database.database import Database
from src.database.models import Repetition
from tests.factories import DAY, T0, add_word, make_reps

F, R = Direction.FORWARD, Direction.REVERSE


class TestLifecycle:
    def test_session_before_init_raises(self):
        with pytest.raises(RuntimeError):
            with Database().session():
                pass

    def test_init_seeds_language_pair_once(self, db):
        init_db(f"sqlite:///{db}", "German", "English")  # re-init must not overwrite
        with get_session() as session:
            lp = LanguagePairRepository(session).get()
        assert (lp.source_language, lp.target_language) == ("French", "Polish")

    def test_session_commits_on_clean_exit(self, db):
        add_word("chat", "kot")
        with get_session() as session:
            assert WordRepository(session).find_by_source_text("chat") is not None

    def test_session_rolls_back_on_exception(self, db):
        with pytest.raises(ZeroDivisionError):
            with get_session() as session:
                session.add(Word(source_text="chien", target_text="pies"))
                session.flush()
                1 / 0
        with get_session() as session:
            assert WordRepository(session).find_by_source_text("chien") is None

    def test_objects_stay_readable_after_session(self, db):
        add_word("chat", "kot")
        with get_session() as session:
            word = WordRepository(session).get_all()[0]
        assert word.source_text == "chat"


def test_migration_adds_ceiling_and_flag_columns(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    tau_cols = ", ".join(
        f"{d}{o}_tau REAL" for d in ("fwd", "rev") for o in ("", "_ok", "_no")
    )
    con.execute(
        f"CREATE TABLE words (id INTEGER PRIMARY KEY, source_text TEXT NOT NULL, target_text TEXT NOT NULL, {tau_cols})"
    )
    con.execute(
        "INSERT INTO words (id, source_text, target_text, fwd_tau) VALUES (1, 'chat', 'kot', 1000.0)"
    )
    con.commit()
    con.close()

    init_db(f"sqlite:///{path}", "French", "Polish")
    init_db(f"sqlite:///{path}", "French", "Polish")  # idempotent

    with get_session() as session:
        word = session.get(Word, 1)
    assert word.fwd_tau == 1000.0
    assert word.fwd_ceiling == 1.0  # seeded where τ was set
    assert word.rev_ceiling is None  # NULL τ keeps a NULL ceiling
    assert word.priority is False and word.ignored is False


class TestWordRepository:
    def test_get_all_ordered_by_id(self, db):
        ids = [add_word(s, t) for s, t in [("b", "1"), ("a", "2"), ("c", "3")]]
        with get_session() as session:
            assert [w.id for w in WordRepository(session).get_all()] == sorted(ids)

    def test_delete_cascades_to_repetitions(self, db):
        wid = add_word("chat", "kot", {F: [(T0, True)], R: [(T0, False)]})
        with get_session() as session:
            repo = WordRepository(session)
            repo.delete(repo.get_by_id(wid))
        with get_session() as session:
            assert session.scalars(select(Repetition)).all() == []


class TestRepetitionRepository:
    def test_get_for_word_filters_direction_and_orders(self, db):
        wid = add_word(
            "chat", "kot", {F: [(T0 + 2, True), (T0, False)], R: [(T0 + 1, True)]}
        )
        with get_session() as session:
            reps = RepetitionRepository(session).get_for_word(wid, F)
        assert [r.practiced_at for r in reps] == [T0, T0 + 2]

    def test_latest_practiced_at(self, db):
        a = add_word("a", "1", {F: [(T0, True), (T0 + 5, True)], R: [(T0 + 9, True)]})
        b = add_word("b", "2")
        with get_session() as session:
            repo = RepetitionRepository(session)
            assert repo.latest_practiced_at_by_word() == {a: T0 + 9}
            assert repo.latest_practiced_at_by_word_direction() == {
                (a, 0): T0 + 5,
                (a, 1): T0 + 9,
            }
        assert b  # a word with no history is simply absent

    def test_history_by_word_direction(self, db):
        a = add_word("a", "1", {F: [(T0 + 1, True), (T0, False)]})
        b = add_word("b", "2", {R: [(T0, True)]})
        c = add_word("c", "3", {F: [(T0, True)]})
        with get_session() as session:
            hist = RepetitionRepository(session).history_by_word_direction([a, b])
        assert set(hist) == {(a, 0), (b, 1)}
        assert [r.practiced_at for r in hist[(a, 0)]] == [T0, T0 + 1]
        assert (c, 0) not in hist

    def test_first_attempt_success_rate(self, db):
        with get_session() as session:
            assert (
                RepetitionRepository(session).first_attempt_success_rate(default=0.42)
                == 0.42
            )
        add_word("a", "1", {F: [(T0, True), (T0 + 1, False)], R: [(T0, False)]})
        add_word(
            "b", "2", {F: [(T0 + 5, False), (T0, True)]}
        )  # earliest is the success
        with get_session() as session:
            assert RepetitionRepository(
                session
            ).first_attempt_success_rate() == pytest.approx(2 / 3)

    def test_count_since_counts_events(self, db):
        add_word("a", "1", {F: [(T0, True), (T0 + DAY, True), (T0 + DAY, False)]})
        with get_session() as session:
            assert RepetitionRepository(session).count_since(T0 + DAY) == 2

    def test_add(self, db):
        wid = add_word("a", "1")
        with get_session() as session:
            RepetitionRepository(session).add(make_reps([(T0, True)], wid)[0])
        with get_session() as session:
            assert len(RepetitionRepository(session).get_for_word(wid, F)) == 1


class TestInspect:
    def test_reads_pair_and_count(self, db):
        add_word("a", "1")
        add_word("b", "2")
        assert read_language_pair(db) == ("French", "Polish")
        assert count_words(db) == 2

    def test_unrelated_file_is_skipped(self, tmp_path):
        junk = tmp_path / "junk.db"
        junk.write_bytes(b"definitely not sqlite")
        assert read_language_pair(junk) is None
        assert count_words(junk) == 0
