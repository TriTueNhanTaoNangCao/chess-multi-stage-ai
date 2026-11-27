"""Run with: python -m streamlit run app.py"""
# Spawned search processes must not execute the Streamlit UI again.
if __name__ != "__mp_main__":
    import os
    from pathlib import Path

    import streamlit as st

    from services.game_controller import GameController
    from services.history import HistoryStore
    from ui.design import apply_design

    st.set_page_config(page_title="Chess Lab · Multi-stage AI", page_icon="♟", layout="wide")
    apply_design()

    if "controller" not in st.session_state:
        default = Path(__file__).parent / "results" / "history.sqlite3"
        st.session_state.controller = GameController(HistoryStore(os.environ.get("CHESS_HISTORY_DB", str(default))))

    page = st.navigation([
        st.Page("app_pages/play.py", title="Bàn đấu", icon=":material/sports_esports:", default=True),
        st.Page("app_pages/tournament.py", title="Giải đấu", icon=":material/emoji_events:"),
        st.Page("app_pages/history.py", title="Lịch sử đối đầu", icon=":material/history:"),
    ], position="top")
    with st.container(key="workspace"):
        page.run()
