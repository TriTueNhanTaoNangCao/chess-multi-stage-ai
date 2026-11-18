"""Chess.com-style 3-column layout: Left (Setup & Eval), Center (Board), Right (Moves & Controls)."""
import json

import chess
import streamlit as st

from agents.stage1_heuristics.pst import evaluate_board
from services.agent_config import AgentConfig
from services.history import standings, csv_text
from ui.board_component import chessboard
from ui.common import REASONS, bot_picker, agent_title, downloads, move_table
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
    st.fragment(_live_match, run_every=0.5 if polling else None)(polling)


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


def _left_setup_panel(controller, game):
    with st.container(border=True, key="left_panel", gap="small"):
        st.html('<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
                '<span class="lab-eyebrow" style="margin:0;">CHESS LAB</span>'
                '<span style="font-size:14px;font-weight:700;color:#eee8dc;">CẤU HÌNH</span></div>')

        # 1. Mode selector
        mode = st.segmented_control("Chế độ", ["Người đấu AI", "Hai bot đấu"], default="Người đấu AI",
                                    key="play_mode", required=True, label_visibility="collapsed")

        # 2. Agent configuration
        if mode == "Người đấu AI":
            color = st.segmented_control("Cầm quân", ["Trắng", "Đen"], default="Trắng",
                                         key="human_color", required=True)
            bot = bot_picker("opponent")
            white, black = (AgentConfig("human"), bot) if color == "Trắng" else (bot, AgentConfig("human"))
        else:
            with st.popover(f"⚪ Trắng · {agent_title(st.session_state.get('white_bot', 'alphabeta'))}", width="stretch"):
                white = bot_picker("white")
            with st.popover(f"⚫ Đen · {agent_title(st.session_state.get('black_bot', 'think3'))}", width="stretch"):
                black = bot_picker("black", "think3")

        # 3. Play execution style: Auto vs Step-by-Step
        play_style = st.segmented_control("Cơ chế", ["⚡ Tự động", "🖱️ Từng bước"],
                                          default="⚡ Tự động", key="play_style", required=True)

        with st.expander("Tùy chọn ván", expanded=False):
            fen = st.text_input("FEN khởi đầu (để trống dùng bàn chuẩn)", key="start_fen")
            max_plies = st.number_input("Giới hạn lượt (ply)", 1, 2000, 300, key="max_plies")

        # 4. Start Game Button
        if st.button("▶ Bắt đầu ván mới", type="primary", key="start", width="stretch"):
            try:
                controller.start(white, black, fen=fen.strip() if fen else "", max_plies=int(max_plies),
                                 mode="play" if mode == "Người đấu AI" else "watch")
                if play_style == "🖱️ Từng bước":
                    controller.running = False
                else:
                    controller.running = True
                st.session_state.flipped = (mode == "Người đấu AI" and color == "Đen")
                st.rerun()
            except Exception as exc:
                st.error(f"Không thể bắt đầu: {exc}")

        st.divider()

        # 5. Live Position Evaluation
        board = game.state.board if game else chess.Board()
        evaluation = evaluate_board(board, check_terminal=False) / 100
        cols_eval = st.columns(2, gap="small")
        cols_eval[0].metric("Lượng giá", f"{evaluation:+.2f}")
        cols_eval[1].metric("Lượt đi", len(game.moves) if game else 0)
        st.caption("Dương: Trắng lợi · Âm: Đen lợi")

        if game and game.telemetry:
            last = game.telemetry[-1]
            val = last.get("nodes_visited", last.get("root_visits"))
            if val is not None:
                st.caption(f"Tìm kiếm: {val:,} nodes/visits")


def _right_move_panel(controller, human):
    game = controller.session
    finished = game is None or game.finished
    can_run = bool(game and (not game.finished or controller.schedule))

    with st.container(border=True, key="right_panel", gap="small"):
        st.caption("📜 BIÊN BẢN NƯỚC ĐI (MOVES)")
        rows = notation_rows(controller.record()) if game else []
        if rows:
            st.dataframe(rows, hide_index=True, height=230,
                         column_config={"#": st.column_config.NumberColumn("#", width="small")})
        else:
            st.html('<div class="empty-moves"><div class="glyph">♞</div>Ván cờ đang chờ.<br>Bấm Bắt đầu để chơi.</div>')

        if game and game.error:
            st.error(game.error)

        # Chess.com Style Control Buttons Bar
        st.caption("ĐIỀU KHIỂN VÁN ĐẤU")
        btn_cols = st.columns(4, gap="small")

        # 1. Undo
        if btn_cols[0].button("↩", help="Lùi lại 1 nước", key="undo",
                              disabled=finished or not game.moves or controller.worker.pending or controller.mode == "tournament",
                              width="stretch"):
            controller.undo()
            st.rerun()

        # 2. Pause / Resume Auto-play
        pause_label = "⏸" if controller.running else "▶"
        pause_help = "Tạm dừng tự động" if controller.running else "Tiếp tục tự động đánh"
        if btn_cols[1].button(pause_label, help=pause_help, key="pause", disabled=not can_run, width="stretch"):
            controller.running = not controller.running
            st.rerun()

        # 3. Next Step (Step-by-step on click!)
        if btn_cols[2].button("⏭", help="Đi tiếp 1 nước theo click của bạn", key="step",
                              disabled=finished or human or controller.running, width="stretch"):
            controller.step_requested = True
            st.rerun()

        # 4. Stop
        if btn_cols[3].button("⏹", help="Dừng ván", key="stop", disabled=not can_run, width="stretch"):
            controller.stop()
            st.rerun()

        # Utilities row: Flip & Options
        sub_cols = st.columns([1.2, 1], gap="small", vertical_alignment="center")
        sub_cols[0].toggle("Xoay bàn", key="flipped")
        with sub_cols[1], st.popover("Tùy chọn", icon=":material/tune:", width="stretch"):
            controller.delay = st.slider("Độ trễ AI (giây)", 0.0, 3.0, 0.4, 0.1, key="move_delay")
            if game and controller.mode == "play" and not game.finished:
                if st.button("🏳️ Đầu hàng", key="resign", width="stretch"):
                    controller.resign(controller.agents[0] is None)
                    st.rerun()
                with st.form("manual_move", clear_on_submit=True):
                    text = st.text_input("Nhập nước (SAN / UCI)", placeholder="e4 hoặc e2e4", key="move_text")
                    if st.form_submit_button("Đi nước", disabled=not human):
                        try:
                            m = game.state.parse_san(text.strip())
                            controller.human_move(m.uci(), controller.token)
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))

        if game:
            with st.expander("Chi tiết & Tải bản ghi", expanded=False):
                move_table(controller.record())
                downloads(controller.record(), "live", disabled=needs_poll(controller))


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

    # 3 Columns Layout: Left (Setup), Center (Board), Right (Moves & Controls)
    col_left, col_center, col_right = st.columns([1.0, 1.85, 1.15], gap="small")

    # Column 1: Left Setup & Eval Panel
    with col_left:
        _left_setup_panel(controller, game)

    # Column 2: Center Board Stage
    with col_center, st.container(key="board_stage", gap="small"):
        board = game.state.board if game else chess.Board()
        flipped = st.session_state.get("flipped", False)
        human = bool(game and not game.finished and controller.agents[game.index] is None)

        if not game:
            status, detail = "Sẵn sàng cho ván mới", "LOCAL ARENA"
        elif game.finished:
            status, detail = f"{game.result} · {REASONS.get(game.reason, game.reason)}", "ĐÃ LƯU KẾT QUẢ"
        else:
            who = "Trắng" if game.state.turn else "Đen"
            state = ("AI đang suy nghĩ" if controller.worker.pending
                     else "Đến lượt bạn" if human
                     else "Đang tự đánh" if controller.running
                     else "Chờ click nước tiếp")
            status = f"{state} · {who}" + (" · Chiếu!" if board.is_check() else "")
            detail = f"LƯỢT {len(game.moves):02d}"

        match_status(status, detail)

        top_index = 0 if flipped else 1
        bottom_index = 1 - top_index

        for index, before in ((top_index, True), (bottom_index, False)):
            if not before:
                event = chessboard(
                    board,
                    key="live_board" if game else "empty_board",
                    token=controller.token if game else "idle",
                    interactive=human,
                    flipped=flipped,
                    caption="Bản ghi đã được lưu" if game and game.finished else "CHESS LAB · THE ART OF THE NEXT MOVE",
                )
                if event and game:
                    try:
                        if controller.human_move(event["uci"], event["token"]):
                            st.rerun()
                    except (ValueError, KeyError) as exc:
                        st.error(str(exc))

            player_strip(
                game.names[index] if game else ("Quân Trắng" if index == 0 else "Quân Đen"),
                white=index == 0,
                seconds=game.times[index] if game else 0,
                active=bool(game and not game.finished and game.index == index),
                ready=game is not None,
            )

    # Column 3: Right Moves & Chess.com Style Controls
    with col_right:
        _right_move_panel(controller, human)

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
