import streamlit as st

from ui.common import bot_picker, agent_title
from ui.live import live_match

controller = st.session_state.controller
with st.container(border=True, key="setup_bar", gap="small"):
    first, second, options, launch = st.columns([1.1, 1.1, 0.85, 1], gap="small", vertical_alignment="center")
    with options, st.popover("Thể thức", icon=":material/tune:", width="stretch", key="tournament_setup"):
        games = st.number_input("Số ván đối đầu (chẵn)", 2, 100, 2, step=2, key="games_per_pair")
        cap = st.number_input("Giới hạn lượt mỗi ván", 1, 2000, 200, key="tournament_cap")
        st.caption("Hai AI đã chọn đánh lần lượt từng ván, đổi màu sau mỗi ván. Hai cấu hình cần khác nhau.")
    with first, st.popover(f"Bot 1 · {agent_title(st.session_state.get('tournament_0_bot', 'alphabeta'))}", width="stretch", key="tournament_white_setup"):
        white = bot_picker("tournament_0")
    with second, st.popover(f"Bot 2 · {agent_title(st.session_state.get('tournament_1_bot', 'random'))}", width="stretch", key="tournament_black_setup"):
        black = bot_picker("tournament_1", "random")
    if launch.button("Bắt đầu giải", type="primary", key="start_tournament", width="stretch"):
        try:
            with st.spinner("Kiểm tra các bot…"):
                controller.start_tournament([white, black], games=int(games), max_plies=int(cap))
            st.rerun()
        except Exception as exc:
            st.error(f"Không thể bắt đầu giải: {exc}")
    st.caption(f"Bot 1 vs Bot 2  ·  {int(games)} ván liên tiếp  ·  Luân phiên Trắng / Đen  ·  Tự lưu kết quả")
live_match()
