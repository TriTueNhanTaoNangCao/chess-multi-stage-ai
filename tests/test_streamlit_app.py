"""Native UI smoke tests; board JavaScript is checked in a real browser."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_start_and_reject_invalid_fen(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_HISTORY_DB", str(tmp_path / "history.sqlite3"))
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    assert not app.exception
    app.text_input(key="start_fen").set_value("invalid fen")
    app.button(key="start").click().run()
    assert not app.exception
    assert app.error
    assert app.session_state.controller.session is None
    app.text_input(key="start_fen").set_value("")
    app.button(key="start").click().run()
    assert not app.exception
    assert app.session_state.controller.session.names[0] == "Người chơi"
    assert len(app.session_state.controller.store.list_games()) == 1


def test_missing_checkpoint_shows_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_HISTORY_DB", str(tmp_path / "history.sqlite3"))
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    app.selectbox(key="opponent_bot").select("supervised").run()
    app.button(key="start").click().run()
    assert not app.exception
    assert app.error
    assert not app.session_state.controller.store.list_games()
