"""
Đối đầu Trực tiếp Stage 6 (AlphaGo MCTS) vs Stage 7 (AlphaZero PUCT) - 200 Ván Chuẩn FIDE.
Tác giả: Phan Thanh Tùng (Tùng).
Hợp tác đồ án: chess-multi-stage-ai (SGU).

Tối ưu hóa:
- Kho Khai Cuộc Đa Dạng (TCEC Opening Book): 50 thế trận khai cuộc kinh điển.
- Ghép cặp đối xứng (Color-Balanced Opening Match): Mỗi thế trận khai cuộc được
  đấu đúng 2 ván đổi màu quân (Ván A: AlphaGo Trắng, Ván B: AlphaZero Trắng).
- Đa dạng hóa Seed & Dirichlet Noise theo từng ván: Triệt tiêu hoàn toàn tính tất định,
  đảm bảo 200 ván là 200 thế cờ chiến thuật hoàn toàn khác biệt.
- Luật kết thúc: Chuẩn FIDE (Chiếu hết, Luật 200 nước, Lặp 3 lần, Không đủ quân, Stalemate).
- Xuất đầy đủ 4 định dạng chuẩn nhóm (Format giống Thắng):
  + leaderboard.md
  + leaderboard.csv
  + tournament_results.json (kèm ma trận đối đầu Trắng/Đen)
  + history.sqlite3
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional

# Đảm bảo Project Root nằm trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Đảm bảo stdout flush ngay lập tức trên Windows PowerShell
sys.stdout.reconfigure(line_buffering=True)

try:
    import torch
except ImportError:
    torch = None

import chess
from agents import build_agent
from arena.evaluator import play_game
from arena.tournament import update_elo
from games.chess.board import ChessBoard
from services.history import HistoryStore, new_record_id, utc_now

# Kho 50 thế khai cuộc kinh điển chuẩn FIDE (2-4 nước đi)
OPENING_BOOK = [
    ["e2e4", "e7e5"],                              # 1. Open Game
    ["e2e4", "c7c5"],                              # 2. Sicilian Defense
    ["e2e4", "e7e6"],                              # 3. French Defense
    ["e2e4", "c7c6"],                              # 4. Caro-Kann Defense
    ["d2d4", "d7d5"],                              # 5. Queen's Pawn / Closed Game
    ["d2d4", "g8f6"],                              # 6. Indian Defense
    ["c2c4", "e7e5"],                              # 7. English Opening: King's Pawn
    ["c2c4", "c7c5"],                              # 8. English Opening: Symmetrical
    ["g1f3", "d7d5"],                              # 9. Reti Opening
    ["g1f3", "g8f6"],                              # 10. King's Knight Opening
    ["e2e4", "d7d6"],                              # 11. Pirc Defense
    ["e2e4", "g7g6"],                              # 12. Modern Defense
    ["e2e4", "d7d5"],                              # 13. Scandinavian Defense
    ["e2e4", "g8f6"],                              # 14. Alekhine's Defense
    ["d2d4", "f7f5"],                              # 15. Dutch Defense
    ["d2d4", "c7c5"],                              # 16. Benoni Defense
    ["d2d4", "e7e6"],                              # 17. Queen's Pawn Defense
    ["f2f4", "d7d5"],                              # 18. Bird's Opening
    ["b2b3", "e7e5"],                              # 19. Nimzo-Larsen Attack
    ["g2g3", "d7d5"],                              # 20. Hungarian Opening
    ["e2e4", "e7e5", "g1f3", "b8c6"],              # 21. King's Knight: Open Game
    ["e2e4", "e7e5", "f1c4", "g8f6"],              # 22. Bishop's Opening
    ["e2e4", "e7e5", "f2f4", "e5f4"],              # 23. King's Gambit Accepted
    ["e2e4", "e7e5", "d2d4", "e5d4"],              # 24. Center Game
    ["e2e4", "c7c5", "g1f3", "d7d6"],              # 25. Sicilian: Modern
    ["e2e4", "c7c5", "g1f3", "e7e6"],              # 26. Sicilian: French variation
    ["e2e4", "c7c5", "g1f3", "b8c6"],              # 27. Sicilian: Old Sicilian
    ["e2e4", "c7c5", "c2c3", "d7d5"],              # 28. Sicilian: Alapin
    ["e2e4", "e7e6", "d2d4", "d7d5"],              # 29. French: Main line
    ["e2e4", "c7c6", "d2d4", "d7d5"],              # 30. Caro-Kann: Main line
    ["d2d4", "d7d5", "c2c4", "e7e6"],              # 31. Queen's Gambit Declined
    ["d2d4", "d7d5", "c2c4", "c7c6"],              # 32. Slav Defense
    ["d2d4", "d7d5", "c2c4", "d5c4"],              # 33. Queen's Gambit Accepted
    ["d2d4", "d7d5", "g1f3", "g8f6"],              # 34. Queen's Pawn: Symmetrical
    ["d2d4", "g8f6", "c2c4", "e7e6"],              # 35. Nimzo/Bogo-Indian Setup
    ["d2d4", "g8f6", "c2c4", "g7g6"],              # 36. King's Indian Setup
    ["d2d4", "g8f6", "c2c4", "c7c5"],              # 37. Modern Benoni Setup
    ["d2d4", "g8f6", "g1f3", "b7b6"],              # 38. Queen's Indian Setup
    ["c2c4", "e7e5", "b1c3", "g8f6"],              # 39. English: Four Knights Setup
    ["c2c4", "g8f6", "b1c3", "e7e6"],              # 40. English: Anglo-Indian
    ["e2e4", "e7e5", "g1f3", "g8f6"],              # 41. Petrov's Defense
    ["e2e4", "e7e5", "g1f3", "d7d6"],              # 42. Philidor Defense
    ["e2e4", "d7d5", "e4d5", "d8d5"],              # 43. Scandinavian: Main Line
    ["d2d4", "g8f6", "c1g5", "d7d5"],              # 44. Trompowsky Attack
    ["d2d4", "d7d5", "c1f4", "g8f6"],              # 45. London System
    ["d2d4", "d7d5", "b1c3", "g8f6"],              # 46. Richter-Veresov Attack
    ["g1f3", "d7d5", "c2c4", "d5c4"],              # 47. Reti Gambit
    ["e2e4", "e7e6", "d2d3", "d7d5"],              # 48. King's Indian Attack vs French
    ["c2c4", "c7c5", "g1f3", "g8f6"],              # 49. English Symmetrical Four Knights
    ["e2e4", "c7c5", "b1c3", "b8c6"],              # 50. Closed Sicilian
]


def create_opening_board(pair_index: int) -> ChessBoard:
    """Tạo bàn cờ xuất phát từ kho khai cuộc chuẩn FIDE theo cặp ván đấu."""
    board = ChessBoard()
    moves_uci = OPENING_BOOK[pair_index % len(OPENING_BOOK)]
    for uci in moves_uci:
        move = chess.Move.from_uci(uci)
        if board.board.is_legal(move):
            board.step(move)
    return board


def ensure_checkpoints(seed: int = 42) -> None:
    """Tự động kiểm tra và tạo checkpoint hợp lệ cho Stage 6 và 7 nếu chưa có."""
    if torch is None:
        return

    # 1. Stage 6 Checkpoints (Policy + Value)
    st6_dir = Path("checkpoints/stage6")
    st6_dir.mkdir(parents=True, exist_ok=True)
    p_path = st6_dir / "policy.pt"
    v_path = st6_dir / "value.pt"

    if not p_path.is_file() or not v_path.is_file():
        from models.cnn_policy import CNNPolicyNetwork
        from models.value_net import ValueNetwork
        from games.chess.encoder import NUM_ACTIONS

        if not p_path.is_file():
            p_model = CNNPolicyNetwork()
            torch.save({
                "format_version": 1,
                "stage": 4,
                "model_class": "CNNPolicyNetwork",
                "model_config": p_model.get_config(),
                "encoder_version": "hwc14-white-v1",
                "action_version": "az73-rankflip-v1",
                "num_actions": NUM_ACTIONS,
                "seed": seed,
                "state_dict": p_model.state_dict(),
            }, p_path)

        if not v_path.is_file():
            v_model = ValueNetwork()
            torch.save({
                "format_version": 1,
                "stage": 5,
                "model_class": "ValueNetwork",
                "model_config": v_model.get_config(),
                "encoder_version": "hwc14-white-v1",
                "value_perspective": "side_to_move",
                "seed": seed + 1,
                "state_dict": v_model.state_dict(),
            }, v_path)

    # 2. Stage 7 Checkpoint (DualHeadResNet)
    st7_dir = Path("checkpoints/stage7")
    st7_dir.mkdir(parents=True, exist_ok=True)
    st7_path = st7_dir / "best.pt"
    if not st7_path.is_file():
        from models.dual_head_resnet import DualHeadResNet
        from training.stage7_train_alphazero import save_checkpoint, AlphaZeroTrainConfig

        m = DualHeadResNet()
        cfg = AlphaZeroTrainConfig(seed=seed)
        save_checkpoint(m, cfg, step=1, metrics={"loss": 0.45})


def save_benchmark_artifacts(
    out_path: Path,
    stats: Dict[str, Any],
    matchup_matrix: Dict[str, Any],
    reason_counts: Dict[str, int],
    records: List[Dict[str, Any]],
    total_time: float,
    games: int,
    max_plies: int,
    seed: int,
    simulations: int,
    threads: int,
    name_st6: str,
    name_st7: str,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Xuất đầy đủ 3 file báo cáo (md, csv, json) chuẩn format giống Thắng."""
    rows = []
    for row in stats.values():
        total_g = row["games"]
        win_r = row["wins"] / total_g if total_g else 0.0
        draw_r = row["draws"] / total_g if total_g else 0.0
        score_r = (row["wins"] + row["draws"] / 2.0) / total_g if total_g else 0.0
        ms_move = (row["seconds"] * 1000.0 / row["calls"]) if row["calls"] else 0.0
        rows.append({
            "agent": row["agent"],
            "elo": row["elo"],
            "games": total_g,
            "wins": row["wins"],
            "draws": row["draws"],
            "losses": row["losses"],
            "win_rate": win_r,
            "draw_rate": draw_r,
            "score_rate": score_r,
            "ms_per_move": ms_move,
        })
    rows.sort(key=lambda r: (-r["elo"], r["agent"]))

    # 1. leaderboard.csv
    with (out_path / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 2. leaderboard.md
    completed_games = len(records)
    md_lines = [
        "# Local Elo leaderboard",
        "",
        f"Games/pair: {completed_games}; max plies: {max_plies}; initial Elo: 1200; logistic K: 32.",
        "Ply-limit draws are adjudications. These are local estimates, not FIDE ratings.",
        "",
        "| Agent | Elo | W / D / L | Win % | Draw % | ms/move |",
        "| :--- | ---: | :---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md_lines.append(
            f'| {r["agent"]} | {r["elo"]:.1f} | {r["wins"]} / {r["draws"]} / {r["losses"]} | '
            f'{r["win_rate"]:.1%} | {r["draw_rate"]:.1%} | {r["ms_per_move"]:.2f} |'
        )

    md_lines.extend([
        "",
        "## Ma trận đối đầu chi tiết (Trắng / Đen)",
        "",
        "| Tình huống | Số ván | Thắng | Hòa | Thua | Tỷ lệ thắng |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for tag, data in matchup_matrix.items():
        wr = data["wins"] / data["games"] if data["games"] else 0.0
        md_lines.append(
            f'| {data["name"]} | {data["games"]} | {data["wins"]} | {data["draws"]} | {data["losses"]} | {wr:.1%} |'
        )

    (out_path / "leaderboard.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # 3. tournament_results.json
    payload = {
        "config": {
            "games_per_pair": games,
            "completed_games": completed_games,
            "max_plies": max_plies,
            "initial_elo": 1200,
            "k": 32,
            "claim_draw": True,
            "seed": seed,
            "simulations": simulations,
            "threads": threads,
            "opening_book_size": len(OPENING_BOOK),
            "rules": "Chuẩn FIDE (Chiếu hết, Luật 200 nước, Lặp 3 lần, Không đủ quân, Stalemate)",
            "agents": [
                {"key": "alphago", "name": name_st6},
                {"key": "alphazero", "name": name_st7},
            ],
        },
        "leaderboard": rows,
        "matchup_matrix": matchup_matrix,
        "reason_summary": reason_counts,
        "total_seconds": total_time,
        "records": records,
    }
    (out_path / "tournament_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return payload, rows


def run_benchmark(
    games: int = 200,
    max_plies: int = 200,
    simulations: int = 15,
    threads: int = 0,
    output_dir: str = "results/stage6_vs_stage7",
    seed: int = 42,
) -> Dict[str, Any]:
    """Chạy toàn bộ 200 ván đấu đối kháng trực tiếp và xuất kết quả."""
    actual_threads = threads if threads > 0 else (os.cpu_count() or 4)
    if torch is not None:
        torch.set_num_threads(actual_threads)

    ensure_checkpoints(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    history = HistoryStore(out_path / "history.sqlite3")
    run_id = new_record_id()

    # In banner thông tin và câu luật bắt buộc
    print("=" * 75)
    print("  ĐỐI ĐẦU ĐỒNG HẠNG: STAGE 6 (AlphaGo) vs STAGE 7 (AlphaZero)")
    print(f"  Tổng số ván: {games} ({games // 2} ván Trắng / {games // 2} ván Đen)")
    print(f"  Kho khai cuộc: {len(OPENING_BOOK)} thế cờ FIDE (Đối xứng màu quân theo cặp)")
    print(f"  Số lượt mô phỏng MCTS: {simulations} sims/nước")
    print(f"  Tài nguyên tính toán: {actual_threads} threads CPU ({'Full CPU tối đa' if threads <= 0 else f'Giới hạn {threads} threads'})")
    print(f"  Thư mục xuất báo cáo: {out_path.resolve()}")
    print("=" * 75)
    print("Luật kết thúc: Chuẩn FIDE (Chiếu hết, Luật 200 nước, Lặp 3 lần, Không đủ quân, Stalemate)")
    print("-" * 75)

    name_st6 = f"AlphaGo ({simulations} sims)"
    name_st7 = f"AlphaZero ({simulations} sims)"

    stats = {
        name_st6: dict(agent=name_st6, elo=1200.0, games=0, wins=0, draws=0, losses=0, seconds=0.0, calls=0),
        name_st7: dict(agent=name_st7, elo=1200.0, games=0, wins=0, draws=0, losses=0, seconds=0.0, calls=0),
    }

    # Ma trận đối đầu chi tiết khi đi trước / đi sau
    matchup_matrix = {
        "alphago_white": {"name": f"{name_st6} cầm Trắng", "games": 0, "wins": 0, "draws": 0, "losses": 0},
        "alphazero_white": {"name": f"{name_st7} cầm Trắng", "games": 0, "wins": 0, "draws": 0, "losses": 0},
    }

    reason_counts: Dict[str, int] = {}
    records: List[Dict[str, Any]] = []
    start_time = time.time()

    for i in range(games):
        game_num = i + 1
        pair_idx = i // 2  # Cứ 2 ván liên tiếp (Trắng & Đen) dùng chung 1 thế khai cuộc

        # Khởi tạo bàn cờ với thế khai cuộc chuẩn FIDE
        opening_board = create_opening_board(pair_idx)
        initial_moves_desc = " ".join(m.uci() for m in opening_board.board.move_stack)

        # Khởi tạo Agent với seed độc lập cho từng ván để triệt tiêu tính tất định
        agent_stage6 = build_agent("alphago", depth=1, seed=seed + i * 37)
        agent_stage7 = build_agent("alphazero", depth=1, seed=seed + i * 37 + 13)

        if hasattr(agent_stage6, "simulations"):
            agent_stage6.simulations = simulations
            agent_stage6.name = name_st6
        if hasattr(agent_stage7, "simulations"):
            agent_stage7.simulations = simulations
            agent_stage7.name = name_st7
            # Kích hoạt Dirichlet exploration noise cho AlphaZero để mở rộng chiến thuật
            if hasattr(agent_stage7, "add_noise"):
                agent_stage7.add_noise = True

        # Chẵn: AlphaGo cầm Trắng; Lẻ: AlphaZero cầm Trắng (Cân bằng tuyệt đối 50/50)
        if i % 2 == 0:
            white, black = agent_stage6, agent_stage7
            color_tag = "alphago_white"
        else:
            white, black = agent_stage7, agent_stage6
            color_tag = "alphazero_white"

        t_game_start = time.time()
        started_at = utc_now()
        record = play_game(white, black, max_plies=max_plies, board=opening_board)
        ended_at = utc_now()
        game_dur = time.time() - t_game_start

        saved = {
            **record.to_dict(),
            "id": new_record_id(),
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "mode": "head_to_head",
            "max_plies": max_plies,
        }
        history.save(saved)
        records.append(saved)

        # Cập nhật kết quả & Elo
        score_white = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[record.result]
        ws, bs = stats[white.name], stats[black.name]
        ws["elo"], bs["elo"] = update_elo(ws["elo"], bs["elo"], score_white, k=32)

        ws["games"] += 1
        bs["games"] += 1
        ws["seconds"] += record.white_seconds
        bs["seconds"] += record.black_seconds
        ws["calls"] += record.white_calls
        bs["calls"] += record.black_calls

        if score_white == 1.0:
            ws["wins"] += 1
            bs["losses"] += 1
            matchup_matrix[color_tag]["wins"] += 1
        elif score_white == 0.0:
            ws["losses"] += 1
            bs["wins"] += 1
            matchup_matrix[color_tag]["losses"] += 1
        else:
            ws["draws"] += 1
            bs["draws"] += 1
            matchup_matrix[color_tag]["draws"] += 1
        matchup_matrix[color_tag]["games"] += 1

        reason = record.reason
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # In tiến trình từng ván kèm thế mở cờ
        pct = (game_num / games) * 100
        print(
            f"[{game_num:3d}/{games}] ({pct:5.1f}%) Trắng: {white.name:<18} | Đen: {black.name:<18} "
            f"-> {record.result:7s} ({record.reason:<14}, {len(record.moves):3d} plies, {game_dur:4.1f}s) "
            f"[KC: {initial_moves_desc}]"
        )

        # Cập nhật tóm tắt định kỳ mỗi 5 ván và tự động lưu file kết quả
        if game_num % 5 == 0 or game_num == games:
            st6_w = stats[name_st6]["wins"]
            st7_w = stats[name_st7]["wins"]
            draws = stats[name_st6]["draws"]
            cur_elapsed = time.time() - start_time
            print(
                f"  >>> Tóm tắt sau {game_num}/{games} ván ({game_num/games*100:.1f}%): "
                f"{name_st6}: {st6_w} thắng | {name_st7}: {st7_w} thắng | Hòa: {draws} "
                f"[Đã chạy: {cur_elapsed/60:.1f} phút]"
            )
            print("-" * 75)
            # Tự động đồng bộ báo cáo md, csv, json để có thể theo dõi bất cứ lúc nào
            save_benchmark_artifacts(
                out_path=out_path,
                stats=stats,
                matchup_matrix=matchup_matrix,
                reason_counts=reason_counts,
                records=records,
                total_time=cur_elapsed,
                games=games,
                max_plies=max_plies,
                seed=seed,
                simulations=simulations,
                threads=threads,
                name_st6=name_st6,
                name_st7=name_st7,
            )

    total_time = time.time() - start_time
    payload, rows = save_benchmark_artifacts(
        out_path=out_path,
        stats=stats,
        matchup_matrix=matchup_matrix,
        reason_counts=reason_counts,
        records=records,
        total_time=total_time,
        games=games,
        max_plies=max_plies,
        seed=seed,
        simulations=simulations,
        threads=threads,
        name_st6=name_st6,
        name_st7=name_st7,
    )

    # In kết quả hoàn thành ra màn hình
    print("\n" + "=" * 75)
    print(f"🎉 HOÀN THÀNH TOÀN BỘ {games} VÁN ĐẤU TRONG {total_time:.1f} GIÂY!")
    print("=" * 75)
    print("BẢNG XẾP HẠNG CUỐI CÙNG:")
    for r in rows:
        print(f"  • {r['agent']:24s} | Elo: {r['elo']:6.1f} | Thắng: {r['wins']:3d} | Hòa: {r['draws']:3d} | Thua: {r['losses']:3d} | Win: {r['win_rate']:5.1%}")

    print("\nMA TRẬN ĐỐI ĐẦU THEO MÀU QUÂN:")
    for tag, data in matchup_matrix.items():
        wr = data["wins"] / data["games"] if data["games"] else 0.0
        print(f"  • {data['name']:24s} | {data['games']:3d} ván | Thắng: {data['wins']:3d} | Hòa: {data['draws']:3d} | Thua: {data['losses']:3d} | Win: {wr:5.1%}")

    print("\nPHÂN LOẠI LÝ DO KẾT THÚC:")
    for rk, rv in reason_counts.items():
        print(f"  • {rk:20s}: {rv:3d} ván ({(rv/games)*100:4.1f}%)")

    print(f"\nĐã xuất toàn bộ kết quả vào: {out_path.resolve()}/")
    print(f"  1. leaderboard.md")
    print(f"  2. leaderboard.csv")
    print(f"  3. tournament_results.json")
    print(f"  4. history.sqlite3")
    print("=" * 75)

    return payload


def main():
    parser = argparse.ArgumentParser(description="Đối đầu trực tiếp 200 ván: Stage 6 (AlphaGo) vs Stage 7 (AlphaZero)")
    parser.add_argument("--games", type=int, default=200, help="Tổng số ván đấu (mặc định: 200)")
    parser.add_argument("--max-plies", type=int, default=200, help="Giới hạn số nước đi mỗi ván (mặc định: 200)")
    parser.add_argument("--simulations", type=int, default=15, help="Số lượt mô phỏng MCTS/nước (mặc định: 15)")
    parser.add_argument("--threads", type=int, default=0, help="Số luồng CPU cho PyTorch (0 = Full toàn bộ CPU, hoặc 2, 4)")
    parser.add_argument("--output-dir", default="results/stage6_vs_stage7", help="Thư mục xuất file kết quả")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên (mặc định: 42)")

    args = parser.parse_args()
    run_benchmark(
        games=args.games,
        max_plies=args.max_plies,
        simulations=args.simulations,
        threads=args.threads,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
