import json

import pandas as pd
import streamlit as st

from services.history import csv_text, game_rows, replay, standings
from ui.board_component import chessboard
from ui.common import REASONS, background_status, downloads, move_table
from ui.design import page_intro

page_intro("Lịch sử đối đầu", "Những ván cờ đã chơi. Những quyết định đáng xem lại.")
background_status()
store = st.session_state.controller.store
with st.expander("Nhập kết quả cũ"):
    st.caption("Nhập tournament_results.json hoặc JSON tải từ một trận. Nhập lại cùng dữ liệu không tạo bản sao.")
    file = st.file_uploader("File kết quả JSON", type=["json"], key="history_upload")
    if st.button("Nhập lịch sử", disabled=file is None, key="import_history") and file is not None:
        try:
            payload = json.loads(file.getvalue())
            if "moves" in payload:
                payload = {"records": [payload]}
            count = store.import_tournament(payload)
            st.success(f"Đã nhập {count} ván mới.")
        except Exception as exc:
            st.error(f"Không thể nhập: {exc}")
if st.button("Làm mới lịch sử", key="refresh_history", icon=":material/refresh:"):
    st.rerun()
records = store.list_games()
if not records:
    st.info("Chưa có lịch sử. Bắt đầu một ván hoặc nhập kết quả từ Arena.")
    st.stop()
names = sorted({r[s] for r in records for s in ("white", "black")})
left, middle, right = st.columns(3)
with left:
    first = st.selectbox("Bot / người chơi A", ["Tất cả"] + names, key="history_a")
with middle:
    second = st.selectbox("Đối thủ B", ["Tất cả"] + names, key="history_b")
with right:
    result_filter = st.selectbox("Kết quả", ["Tất cả", "Trắng thắng", "Đen thắng", "Hòa", "Chưa có kết quả"], key="history_result")
runs = sorted({r.get("run_id", "") for r in records})
run = st.selectbox("Đợt thi đấu", ["Tất cả"] + runs, key="history_run")
filtered = []
mapping = {"Trắng thắng": "1-0", "Đen thắng": "0-1", "Hòa": "1/2-1/2", "Chưa có kết quả": "*"}
for record in records:
    sides = [record["white"], record["black"]]
    if first != "Tất cả" and first not in sides:
        continue
    if second != "Tất cả" and second not in sides:
        continue
    if first == second != "Tất cả" and sides != [first, first]:
        continue
    if result_filter != "Tất cả" and record["result"] != mapping[result_filter]:
        continue
    if run != "Tất cả" and record.get("run_id", "") != run:
        continue
    filtered.append(record)
done = [r for r in filtered if r["result"] != "*"]
metrics = st.columns(4)
metrics[0].metric("Tổng số ván", len(filtered))
metrics[1].metric("Đã phân định", len(done))
metrics[2].metric("Hòa", sum(r["result"] == "1/2-1/2" for r in done))
metrics[3].metric("Chưa hoàn tất / dừng", len(filtered)-len(done))
if first != "Tất cả" and second != "Tất cả" and first != second:
    wins = sum((r["white"] == first and r["result"] == "1-0") or
               (r["black"] == first and r["result"] == "0-1") for r in done)
    draws = sum(r["result"] == "1/2-1/2" for r in done)
    st.markdown(f"**{first} đối đầu {second}: {wins} thắng · {draws} hòa · {len(done)-wins-draws} thua**")
rows = standings(filtered)
if rows:
    st.subheader("Thành tích trong phạm vi đang lọc")
    st.dataframe(rows, hide_index=True, column_config={
        "agent": "Bot", "variant": "Mã cấu hình", "games": "Ván", "wins": "Thắng",
        "draws": "Hòa", "losses": "Thua", "win_rate": "Thắng (%)", "ms_per_move": "ms / nước", "elo": "Elo nội bộ"})
    st.caption("Elo khởi tạo 1200, tính lại trên các ván đã lọc; không phải Elo FIDE. Cùng thuật toán nhưng khác depth/model là cấu hình riêng.")
    plot = pd.DataFrame(rows).rename(columns={"wins": "Thắng", "draws": "Hòa", "losses": "Thua"})
    plot["Bot"] = plot["agent"] + " · " + plot["variant"].str[:6]
    st.bar_chart(plot.set_index("Bot"), y=["Thắng", "Hòa", "Thua"],
                 color=["#82A996", "#C9A96E", "#C77973"], horizontal=True, x_label="Số ván")
st.subheader("Danh sách ván")
st.dataframe(game_rows(filtered), hide_index=True, column_config={
    "id": "Mã trận", "started_at": "Bắt đầu (UTC)", "white": "Trắng", "black": "Đen",
    "result": "Kết quả", "winner": "Bên thắng", "reason": "Lý do", "plies": "Lượt",
    "mode": "Chế độ", "run_id": "Đợt đấu", "white_seconds": "Trắng (s)", "black_seconds": "Đen (s)"})
a, b = st.columns(2)
a.download_button("Xuất danh sách CSV", csv_text(game_rows(filtered)), "matches.csv", "text/csv", key="export_csv", on_click="ignore")
b.download_button("Xuất toàn bộ dữ liệu đang lọc", json.dumps({"records": filtered}, ensure_ascii=False, indent=2),
                  "matches.json", "application/json", key="export_json", on_click="ignore")
if not filtered:
    st.info("Không có ván phù hợp bộ lọc.")
    st.stop()
by_id = {r["id"]: r for r in filtered}
selected = st.selectbox("Chọn ván để phát lại", list(by_id), key="replay_game",
                         format_func=lambda key: f"{by_id[key]['started_at']} · {by_id[key]['white']} — {by_id[key]['black']} · {by_id[key]['result']} · {key[:8]}")
record = by_id[selected]
st.subheader(f"{record['result']} · {REASONS.get(record.get('reason', ''), record.get('reason', ''))}")
left, right = st.columns([1.35, 1], gap="large")
with left:
    total = len(record["moves"])
    ply = st.slider("Phát lại đến lượt", 0, total, total, key=f"replay_ply_{selected}") if total else 0
    board = replay(record, ply)
    flipped = st.toggle("Xoay bàn phát lại", key="replay_flipped")
    chessboard(board.board, key="replay_board", token=f"{selected}:{ply}", flipped=flipped,
               caption=f"Lượt {ply}/{total}" + (f" · {record['moves'][ply-1]}" if ply else " · Thế cờ ban đầu"))
with right:
    move_table(record)
    if record.get("error"):
        st.error(record["error"])
    if record.get("import_note"):
        st.caption(record["import_note"])
    downloads(record, "history")
    with st.expander("Chi tiết và cấu hình"):
        st.code(board.fen(), language=None)
        st.json({k: v for k, v in record.items() if k not in ("moves", "telemetry")})
