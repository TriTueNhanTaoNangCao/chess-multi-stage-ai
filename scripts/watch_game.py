"""
Tool to watch two AI agents play against each other (GUI or Terminal)
and optionally demonstrate AlphaZero self-play learning.

Usage:
  # Watch AlphaZero vs Random in Pygame GUI:
  python scripts/watch_game.py --white alphazero --black random --gui --delay 0.5

  # Watch AlphaZero self-play (AlphaZero vs AlphaZero) in Terminal:
  python scripts/watch_game.py --white alphazero --black alphazero --cli --delay 0.3

  # Watch self-play and perform an online learning step (Self-play & Learn):
  python scripts/watch_game.py --white alphazero --black alphazero --learn
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import chess

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import build_agent, AGENT_REGISTRY
from games.chess.board import ChessBoard
from games.chess.renderer import ChessRenderer


def play_in_terminal(
    agent_white_key: str,
    agent_black_key: str,
    depth: int = 2,
    seed: int = 42,
    delay: float = 0.5,
    max_plies: int = 100,
):
    """Watch two bots play chess in terminal with clean ASCII board visualization."""
    board = ChessBoard()
    white_bot = build_agent(agent_white_key, depth=depth, seed=seed)
    black_bot = build_agent(agent_black_key, depth=depth, seed=seed + 1)

    print("=" * 60)
    print(f"TRẬN ĐẤU CỜ VUA: {white_bot.name} (TRẮNG) VS {black_bot.name} (ĐEN)")
    print("=" * 60)

    ply = 0
    while not board.is_game_over() and ply < max_plies:
        is_white_turn = board.turn == chess.WHITE
        current_bot = white_bot if is_white_turn else black_bot
        turn_str = "TRẮNG" if is_white_turn else "ĐEN"

        print(f"\n--- Nước {ply + 1} | Lượt đi: {turn_str} ({current_bot.name}) ---")
        print(ChessRenderer.render(board.board, flipped=not is_white_turn, ascii_only=True))

        t0 = time.perf_counter()
        move = current_bot.act(board.clone())
        dt = time.perf_counter() - t0

        san = board.san(move)
        board.step(move)
        ply += 1

        print(f">> {current_bot.name} đi nước: {san} (UCI: {move.uci()}) - Thời gian tính: {dt:.2f}s")
        if delay > 0:
            time.sleep(delay)

    print("\n" + "=" * 60)
    print("KẾT THÚC VÁN ĐẤU!")
    print("=" * 60)
    print(ChessRenderer.render(board.board, flipped=False, ascii_only=True))
    outcome = board.outcome()
    if outcome is not None:
        winner = "HÒA"
        if outcome.winner == chess.WHITE:
            winner = f"TRẮNG ({white_bot.name}) THẮNG"
        elif outcome.winner == chess.BLACK:
            winner = f"ĐEN ({black_bot.name}) THẮNG"
        print(f"Kết quả: {outcome.result()} | Người thắng: {winner} | Lý do: {outcome.termination.name}")
    else:
        print(f"Ván đấu kết thúc do chạm giới hạn {max_plies} plies (Adjudication).")


def play_and_learn_alphazero(simulations: int = 15, max_plies: int = 60):
    """Demonstrate AlphaZero closed-loop self-play and online learning."""
    import torch
    from models.dual_head_resnet import DualHeadResNet
    from training.stage7_train_alphazero import (
        AlphaZeroTrainConfig,
        AlphaZeroReplayBuffer,
        play_self_play_game,
        train_step,
    )

    print("=" * 65)
    print("TIẾN TRÌNH ALPHAZERO TỰ CHƠI VÀ TỰ HỌC (SELF-PLAY & LEARNING)")
    print("=" * 65)
    print("1. Khởi tạo mạng nơ-ron hai đầu DualHeadResNet (Policy Head + Value Head)...")
    model = DualHeadResNet(num_res_blocks=2, hidden_dim=64, policy_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer = AlphaZeroReplayBuffer(capacity=500)

    cfg = AlphaZeroTrainConfig(
        mcts_simulations=simulations,
        max_plies_per_game=max_plies,
        batch_size=8,
    )

    print(f"2. Bắt đầu ván cờ Tự chơi (Self-play) với MCTS ({simulations} simulations/nước)...")
    t0 = time.time()
    samples = play_self_play_game(model, cfg, seed=42)
    dt_play = time.time() - t0
    buffer.extend(samples)

    print(f"   -> Hoàn thành ván tự chơi trong {dt_play:.1f}s!")
    print(f"   -> Thu thập được {len(samples)} thế cờ và phân phối xác suất MCTS pi.")

    print("3. Thực hiện bước Tự học (Gradient descent trên Replay Buffer)...")
    metrics = train_step(model, optimizer, buffer, cfg)
    print("   -> Cập nhật mạng nơ-ron thành công!")
    print(f"   -> Tổng Loss: {metrics['loss']:.4f}")
    print(f"   -> Policy Loss (Cross-Entropy): {metrics['policy_loss']:.4f}")
    print(f"   -> Value Loss (MSE): {metrics['value_loss']:.4f}")
    print("=" * 65)
    print("AlphaZero đã hoàn tất 1 chu trình tự chơi và tự tối ưu não bộ!")


def main():
    parser = argparse.ArgumentParser(description="Watch 2 Chess Bots play or AlphaZero self-learn")
    parser.add_argument("--white", default="alphazero", choices=AGENT_REGISTRY.keys(), help="Agent playing White")
    parser.add_argument("--black", default="random", choices=AGENT_REGISTRY.keys(), help="Agent playing Black")
    parser.add_argument("--depth", type=int, default=2, help="Search depth / simulation budget")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay in seconds between moves")
    parser.add_argument("--max-plies", type=int, default=60, help="Maximum plies before adjudication")
    parser.add_argument("--learn", action="store_true", help="Run AlphaZero self-play and training step")
    args = parser.parse_args()

    if args.learn:
        play_and_learn_alphazero()
    else:
        play_in_terminal(
            agent_white_key=args.white,
            agent_black_key=args.black,
            depth=args.depth,
            delay=args.delay,
            max_plies=args.max_plies,
        )


if __name__ == "__main__":
    main()
