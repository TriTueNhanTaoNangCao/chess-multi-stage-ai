"""Centered board with compact analysis and move-list panels."""
import json

import chess
import streamlit as st

from agents.stage1_heuristics.pst import evaluate_board
from services.history import standings, csv_text
from ui.board_component import chessboard
from ui.common import REASONS, downloads, move_table
from ui.design import player_strip, match_status


def needs_poll(controller):
    game = controller.session
    if game is None:
        return False
    if game.finished:
        return bool(controller.schedule and controller.running) or controller.worker.pending
    if not (controller.running or controller.step_requested):
        return False
    return controller.worker.pending or controller.agents[game.index] is not None


def live_match():
    polling = needs_poll(st.session_state.controller)
    st.fragment(_live_match, run_every=0.6 if polling else None)(polling)


def notation_rows(record):
    board = chess.Board(record["initial_fen"])
    rows = []
    for uci in record["moves"]:
        move = chess.Move.from_uci(uci)
        if not rows or rows[-1]["#"] != board.fullmove_number:
            rows.append({"#": board.fullmove_number, "Trắng": "—", "Đen": "—"})
        rows[-1]["Trắng" if board.turn else "Đen"] = board.san(move)
        board.push(move)
    return rows


def _analysis_panel(game):
    with st.container(border=True, key="left_panel", gap="small"):
        st.caption("PHÂN TÍCH VÁN")
        board = game.state.board if game else chess.Board()
        evaluation = evaluate_board(board, check_terminal=False) / 100
        st.metric("Lượng giá · phía Trắng", f"{evaluation:+.2f}", help="Đơn vị tốt; lượng giá tĩnh vật chất và vị trí, không phải xác suất thắng.")
        st.caption("Dương lợi Trắng · Âm lợi Đen")
        st.metric("Lượt đã đi", len(game.moves) if game else 0)
        st.caption(f"Giới hạn {game.max_plies} lượt" if game else "Chưa bắt đầu ván đấu")
        last = game.telemetry[-1] if game and game.telemetry else {}
        value = last.get("nodes_visited", last.get("root_visits"))
        st.metric("Nodes / visits gần nhất", f"{value:,}" if value is not None else "—")
        if last.get("cutoffs") is not None:
            st.caption(f"{last['cutoffs']:,} lần cắt nhánh")
        if last.get("simulation_budget") is not None:
            st.caption(f"Ngân sách: {last['simulation_budget']:,} mô phỏng")
        st.html('<div class="side-note">Mỗi nước đi là một quyết định.<br>Theo dõi cách từng thế hệ AI tìm kiếm.</div>')


def _move_panel(controller):
    game = controller.session
    with st.container(border=True, key="right_panel", gap="small"):
        st.caption("BIÊN BẢN NƯỚC ĐI")
        rows = notation_rows(controller.record()) if game else []
        if rows:
            st.dataframe(rows, hide_index=True, height=260,
                         column_config={"#": st.column_config.NumberColumn("#", width="small")})
        else:
            st.html('<div class="empty-moves"><div class="glyph">♞</div>Ván cờ đang chờ bạn.<br>Chọn cấu hình phía trên để bắt đầu.</div>')
        if not game:
            return
        if game.error:
            st.error(game.error)
        with st.expander("Thời gian & tìm kiếm"):
            st.caption("Chỉ tính thời gian AI suy nghĩ, không gồm khoảng nghỉ. Thống kê tùy theo bot hỗ trợ.")
            move_table(controller.record())
        downloads(controller.record(), "live", disabled=needs_poll(controller))
        if needs_poll(controller):
            st.caption("Tạm dừng để tải bản ghi hiện tại.")
        st.caption(f"Đã lưu · {controller.id[:8]}")


def _controls(controller, human):
    game = controller.session
    finished = game is None or game.finished
    can_run = bool(game and (not game.finished or controller.schedule))
    controls = st.columns(4, gap="small")
    if controls[0].button("Tạm dừng" if controller.running else "Tiếp tục", key="pause",
                          disabled=not can_run, width="stretch"):
        controller.running = not controller.running
        st.rerun()
    if controls[1].button("Một nước", key="step", disabled=finished or human or controller.running,
                          width="stretch"):
        controller.step_requested = True
        st.rerun()
    if controls[2].button("Lùi nước", key="undo", disabled=finished or not game.moves or
                          controller.worker.pending or controller.mode == "tournament", width="stretch"):
        controller.undo()
        st.rerun()
    if controls[3].button("Dừng", key="stop", disabled=not can_run, width="stretch"):
        controller.stop()
        st.rerun()
    with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
        st.toggle("Xoay bàn cờ", key="flipped")
        with st.popover("Tùy chọn bàn", icon=":material/tune:"):
            controller.delay = st.slider("Khoảng nghỉ giữa các nước (giây)", 0.0, 3.0, 0.4, 0.1, key="move_delay")
            if game:
                st.caption("FEN hiện tại")
                st.code(game.state.fen(), language=None)
                with st.expander("Cấu hình hai bên"):
                    st.json(controller.configs)
            st.caption("Giữ tab mở khi cho bot tự đánh. Dừng ván bỏ kết quả đang tính; không cưỡng chế dừng tác vụ nền.")
        if game and controller.mode == "play" and not game.finished:
            with st.popover("Nhập nước", icon=":material/keyboard:"):
                with st.form("manual_move", clear_on_submit=True):
                    text = st.text_input("Nhập SAN / UCI", placeholder="e4 hoặc e2e4", key="move_text")
                    if st.form_submit_button("Đi nước", disabled=not human):
                        try:
                            move = game.state.parse_san(text.strip())
                            controller.human_move(move.uci(), controller.token)
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
                if st.button("Đầu hàng", key="resign", width="stretch"):
                    controller.resign(controller.agents[0] is None)
                    st.rerun()


def _live_match(polling):
    controller = st.session_state.controller
    try:
        controller.tick()
    except Exception as exc:
        controller.running = False
        st.error(f"Không thể cập nhật ván: {exc}")
        return
    if polling != needs_poll(controller):
        st.rerun()
    game = controller.session
    if controller.last_error:
        st.error(controller.last_error)
    if game and controller.mode == "tournament":
        complete = controller.schedule_done + int(game.finished)
        st.progress(complete / controller.schedule_total,
                    text=f"Giải đấu · {complete}/{controller.schedule_total} ván hoàn tất")
    left, center, right = st.columns([1, 2.5, 1], gap="medium")
    with left:
        _analysis_panel(game)
    with center, st.container(key="board_stage", gap="small"):
        board = game.state.board if game else chess.Board()
        flipped = st.session_state.get("flipped", False)
        human = bool(game and not game.finished and controller.agents[game.index] is None)
        if not game:
            status, detail = "Sẵn sàng cho ván mới", "LOCAL ARENA"
        elif game.finished:
            status, detail = f"{game.result} · {REASONS.get(game.reason, game.reason)}", "ĐÃ LƯU KẾT QUẢ"
        else:
            who = "Trắng" if game.state.turn else "Đen"
            state = "AI đang suy nghĩ" if controller.worker.pending else "Đến lượt bạn" if human else "Đang thi đấu" if controller.running else "Tạm dừng"
            status = f"{state} · {who}" + (" · Chiếu" if board.is_check() else "")
            detail = f"LƯỢT {len(game.moves):02d}"
        match_status(status, detail)
        top_index = 0 if flipped else 1
        bottom_index = 1 - top_index
        for index, before in ((top_index, True), (bottom_index, False)):
            if not before:
                event = chessboard(board, key="live_board" if game else "empty_board",
                                   token=controller.token if game else "idle", interactive=human, flipped=flipped,
                                   caption="Bản ghi đã được lưu" if game and game.finished else "CHESS LAB · THE ART OF THE NEXT MOVE")
                if event and game:
                    try:
                        if controller.human_move(event["uci"], event["token"]):
                            st.rerun()
                    except (ValueError, KeyError) as exc:
                        st.error(str(exc))
            player_strip(game.names[index] if game else ("Quân Trắng" if index == 0 else "Quân Đen"),
                         white=index == 0, seconds=game.times[index] if game else 0,
                         active=bool(game and not game.finished and game.index == index), ready=game is not None)
        _controls(controller, human)
    with right:
        _move_panel(controller)
    if game and controller.mode == "tournament":
        records = [r for r in controller.store.list_games() if r["run_id"] == controller.run_id]
        rows = standings(records)
        if rows:
            with st.expander("Bảng xếp hạng giải hiện tại", expanded=game.finished and not controller.schedule):
                st.dataframe(rows, hide_index=True)
                with st.container(horizontal=True):
                    st.download_button("Tải bảng xếp hạng CSV", csv_text(rows), "leaderboard.csv", "text/csv",
                                       key="tournament_csv", on_click="ignore", disabled=needs_poll(controller))
                    st.download_button("Tải dữ liệu giải JSON", json.dumps({"records": records}, ensure_ascii=False, indent=2),
                                       "tournament_results.json", "application/json", key="tournament_json",
                                       on_click="ignore", disabled=needs_poll(controller))
