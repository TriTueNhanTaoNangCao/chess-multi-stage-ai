import json

import pandas as pd
import streamlit as st

from agents import AGENT_REGISTRY
from services.agent_config import AgentConfig
from services.history import to_pgn

REASONS = {
    "checkmate": "Chiếu hết", "stalemate": "Hết nước đi hợp lệ",
    "insufficient_material": "Không đủ quân chiếu hết", "threefold_repetition": "Lặp lại 3 lần",
    "fivefold_repetition": "Lặp lại 5 lần", "fifty_moves": "Luật 50 nước",
    "seventyfive_moves": "Luật 75 nước", "max_plies_adjudication": "Hòa do giới hạn lượt",
    "forfeit": "Bot lỗi / đi sai luật", "reset_forfeit": "Bot lỗi khởi tạo",
    "double_forfeit": "Cả hai bot lỗi khởi tạo", "aborted": "Đã dừng, không tính kết quả",
    "resignation": "Đầu hàng", "": "Chưa hoàn tất",
}


def agent_title(key):
    return {"random": "Random", "think1": "Think 1", "think2": "Think 2", "think3": "Think 3",
            "minimax": "Minimax", "alphabeta": "AlphaBeta", "mcts_pure": "Pure MCTS",
            "supervised": "Supervised", "rl_policy": "RL Policy", "alphago": "AlphaGo",
            "alphazero": "AlphaZero"}.get(key, key)


def bot_picker(prefix, default="alphabeta"):
    keys = list(AGENT_REGISTRY)
    key = st.selectbox("Thuật toán", keys, index=keys.index(default), key=f"{prefix}_bot")
    depth, simulations, checkpoint, value, device = 2, 100, "", "", "cpu"
    if key in ("minimax", "alphabeta"):
        depth = st.slider("Độ sâu", 1, 6, 2, key=f"{prefix}_depth")
        st.caption("Depth cao có thể mất nhiều thời gian, nhất là Minimax.")
    if key in ("mcts_pure", "alphago", "alphazero"):
        simulations = st.number_input("Mô phỏng / nước", 1, 10000, 100, step=25, key=f"{prefix}_sims")
    if key in ("supervised", "rl_policy", "alphago", "alphazero"):
        checkpoint = st.text_input("Checkpoint local (.pt)", key=f"{prefix}_checkpoint",
                                   placeholder="checkpoints/stage4/best.pt").strip()
        if key == "alphago":
            value = st.text_input("Value checkpoint (.pt)", key=f"{prefix}_value").strip()
        device = st.selectbox("Thiết bị", ["cpu", "cuda"], key=f"{prefix}_device")
        if not checkpoint or (key == "alphago" and not value):
            st.caption("Stage 4–5 cần checkpoint. Stage 6–7 thiếu trọng số sẽ dùng mạng chưa huấn luyện.")
    with st.expander("Nâng cao"):
        seed = st.number_input("Seed", 0, 2147483647, 42, key=f"{prefix}_seed")
    return AgentConfig(key, int(depth), int(simulations), int(seed), checkpoint, value, device)


def downloads(record, prefix, *, disabled=False):
    a, b = st.columns(2)
    a.download_button("Tải PGN", to_pgn(record), file_name=f"{record['id']}.pgn",
                      mime="application/x-chess-pgn", key=f"{prefix}_pgn", width="stretch", on_click="ignore", disabled=disabled)
    b.download_button("Tải JSON", json.dumps(record, ensure_ascii=False, indent=2),
                      file_name=f"{record['id']}.json", mime="application/json",
                      key=f"{prefix}_json", width="stretch", on_click="ignore", disabled=disabled)


def move_table(record):
    data = record.get("telemetry", [])
    if not data:
        if record.get("moves"):
            st.code(" ".join(record["moves"]), language=None)
            st.caption("Bản ghi cũ không có thời gian từng nước.")
        else:
            st.caption("Chưa có nước đi.")
        return
    rows = []
    for entry in data:
        rows.append({"Lượt": entry["ply"], "Bên": "Trắng" if entry["color"] == "white" else "Đen",
                     "Nước": entry.get("san", "Lỗi"), "Giây": round(entry["seconds"], 3),
                     "Nodes": entry.get("nodes_visited"), "Cắt nhánh": entry.get("cutoffs"),
                     "Visits gốc": entry.get("root_visits"), "Ngân sách mô phỏng": entry.get("simulation_budget")})
    st.dataframe(rows, hide_index=True, height=260)
    if len(rows) > 1:
        frame = pd.DataFrame(rows).set_index("Lượt")
        st.line_chart(frame, y="Giây", height=150, color="#C9A96E")


@st.fragment(run_every=0.5)
def background_status():
    controller = st.session_state.controller
    try:
        controller.tick()
    except Exception as exc:
        controller.running = False
        st.error(str(exc))
    if controller.session and not controller.session.finished:
        st.caption(f"Ván hiện tại: {len(controller.session.moves)} lượt · quay lại Bàn đấu để điều khiển.")
