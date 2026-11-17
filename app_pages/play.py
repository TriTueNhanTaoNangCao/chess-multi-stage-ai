import streamlit as st

from services.agent_config import AgentConfig
from ui.common import bot_picker, agent_title
from ui.live import live_match
from ui.design import page_intro

page_intro("Bàn đấu", "Một bàn cờ. Bảy thế hệ trí tuệ. Nước đi tiếp theo thuộc về bạn.")
mode = st.segmented_control("Chế độ", ["Người đấu AI", "Hai bot đấu"], default="Người đấu AI",
                            key="play_mode", required=True, label_visibility="collapsed")
controller = st.session_state.controller
with st.container(border=True, key="setup_bar", gap="small"):
    first, second, options, launch = st.columns([1.1, 1.1, 0.85, 1], gap="small", vertical_alignment="center")
    if mode == "Người đấu AI":
        with first, st.popover("Người chơi", icon=":material/person:", width="stretch", key="human_setup"):
            color = st.segmented_control("Cầm quân", ["Trắng", "Đen"], default="Trắng", key="human_color", required=True)
            st.caption("Kéo thả quân hoặc click quân rồi chọn ô đích.")
        bot_label = agent_title(st.session_state.get("opponent_bot", "alphabeta"))
        with second, st.popover(f"AI · {bot_label}", width="stretch", key="opponent_setup"):
            bot = bot_picker("opponent")
        white, black = (AgentConfig("human"), bot) if color == "Trắng" else (bot, AgentConfig("human"))
    else:
        with first, st.popover(f"Trắng · {agent_title(st.session_state.get('white_bot', 'alphabeta'))}", width="stretch", key="white_setup"):
            white = bot_picker("white")
        with second, st.popover(f"Đen · {agent_title(st.session_state.get('black_bot', 'think3'))}", width="stretch", key="black_setup"):
            black = bot_picker("black", "think3")
    with options, st.popover("Thiết lập", icon=":material/tune:", width="stretch", key="game_setup"):
        fen = st.text_input("FEN khởi đầu (để trống dùng bàn chuẩn)", key="start_fen")
        max_plies = st.number_input("Giới hạn lượt (ply)", 1, 2000, 300, key="max_plies")
        st.caption("Cấu hình áp dụng cho ván mới. Ván hiện tại được giữ cho đến khi bấm Bắt đầu.")
    if launch.button("Bắt đầu ván mới", type="primary", key="start", width="stretch"):
        try:
            with st.spinner("Đang chuẩn bị hai bên…"):
                controller.start(white, black, fen=fen.strip(), max_plies=int(max_plies),
                                 mode="play" if mode == "Người đấu AI" else "watch")
                st.session_state.flipped = mode == "Người đấu AI" and color == "Đen"
            st.rerun()
        except Exception as exc:
            st.error(f"Không thể bắt đầu: {exc}")
live_match()
