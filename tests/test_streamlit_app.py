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



def test_time_control_toggle_and_real_timed_move(tmp_path, monkeypatch):
    import time
    monkeypatch.setenv("CHESS_HISTORY_DB", str(tmp_path / "history.sqlite3"))
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    assert not app.toggle(key="opponent_timed").value
    app.toggle(key="opponent_timed").set_value(True).run()
    assert app.number_input(key="opponent_move_seconds").value == 10.0
    app.number_input(key="opponent_move_seconds").set_value(0.2)
    app.slider(key="opponent_depth").set_value(6)
    app.button(key="start").click().run()
    assert not app.exception and not app.error
    controller = app.session_state.controller
    try:
        assert controller.config_objects[1].move_time_limit_seconds == 0.2
        controller.human_move("e2e4", controller.token)
        deadline = time.monotonic() + 15
        while len(controller.session.moves) < 2:
            assert time.monotonic() < deadline
            controller.tick()
            time.sleep(0.01)
        entry = controller.record()["telemetry"][-1]
        assert entry["time_limit_seconds"] == 0.2 and not entry.get("error")
        assert entry["time_exhausted"]
        app.toggle(key="opponent_timed").set_value(False).run()
        app.button(key="start").click().run()
        assert not app.exception
        assert app.session_state.controller.config_objects[1].move_time_limit_seconds is None
    finally:
        app.session_state.controller.stop()



def test_tournament_has_one_setup_and_play_keeps_its_own_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_HISTORY_DB", str(tmp_path / "history.sqlite3"))
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    assert app.button(key="start")
    app.switch_page("app_pages/tournament.py").run()
    assert not app.exception
    assert app.button(key="start_tournament")
    assert "start" not in {w.key for w in app.button}
    assert "play_mode" not in {w.key for w in app.segmented_control}
    assert {"tournament_0_bot", "tournament_1_bot"} <= {w.key for w in app.selectbox}
    assert {"tournament_0_timed", "tournament_1_timed"} <= {w.key for w in app.toggle}
    assert not app.session_state.controller.session
    app.switch_page("app_pages/play.py").run()
    assert not app.exception
    assert app.button(key="start")
    assert "start_tournament" not in {w.key for w in app.button}


def test_tournament_readonly_panel_uses_active_players_and_swaps_colors(tmp_path, monkeypatch):
    import time
    monkeypatch.setenv("CHESS_HISTORY_DB", str(tmp_path / "history.sqlite3"))
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    app.switch_page("app_pages/tournament.py").run()
    app.selectbox(key="tournament_0_bot").select("random")
    app.selectbox(key="tournament_1_bot").select("think1")
    app.number_input(key="tournament_cap").set_value(2)
    app.button(key="start_tournament").click().run()
    assert not app.exception and not app.error
    controller = app.session_state.controller
    controller.running = False
    original = controller.id, controller.run_id, list(controller.session.names)
    try:
        app.selectbox(key="tournament_1_bot").select("think3").run()
        assert not app.exception
        assert (controller.id, controller.run_id, list(controller.session.names)) == original
        assert controller.mode == "tournament"
        assert [c.key for c in controller.config_objects] == ["random", "think1"]
        panel = app.main.get("markdown")
        assert any(m.value == "**\u0110en \u00b7 Think 1**" for m in panel)
        assert "start" not in {w.key for w in app.button}
        controller.delay = 0
        controller.running = True
        deadline = time.monotonic() + 10
        while not (controller.session.finished and not controller.schedule):
            assert time.monotonic() < deadline
            controller.tick()
            time.sleep(0.005)
        app.run()
        assert not app.exception
        assert [c.key for c in controller.config_objects] == ["think1", "random"]
        assert any(m.value == "**Tr\u1eafng \u00b7 Think 1**" for m in app.main.get("markdown"))
        records = controller.store.list_games()
        assert len(records) == 2
        assert records[0]["white"] == records[1]["black"]
        assert len({r["run_id"] for r in records}) == 1
    finally:
        controller.stop()


def test_opening_tournament_page_does_not_replace_a_single_game(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_HISTORY_DB", str(tmp_path / "history.sqlite3"))
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    app.button(key="start").click().run()
    controller = app.session_state.controller
    before = controller.id, controller.run_id
    try:
        app.switch_page("app_pages/tournament.py").run()
        assert not app.exception
        assert (controller.id, controller.run_id) == before
        assert "start" not in {w.key for w in app.button}
        assert any("V\u00e1n l\u1ebb \u0111ang m\u1edf" in item.value for item in app.caption)
    finally:
        controller.stop()
