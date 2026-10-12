"""Single GUI/terminal entry point."""
import argparse
import sys
import chess
from agents import AGENT_REGISTRY, build_agent, HumanAgent
from games.chess.board import ChessBoard


def needs_ascii(stream):
    try:
        "♔♚".encode(getattr(stream, "encoding", None) or "utf-8")
        return False
    except (UnicodeEncodeError, LookupError):
        return True


def run_cli(args):
    from games.chess.renderer import ChessRenderer
    from colorama import just_fix_windows_console
    from time import perf_counter
    just_fix_windows_console()
    board = ChessBoard(args.fen)
    bot = build_agent(args.agent, args.depth, args.seed)
    human = HumanAgent()
    human_color = args.side == "white"
    ascii_only = args.ascii or needs_ascii(sys.stdout)
    try:
        while True:
            print(ChessRenderer.render(board.board, flipped=not human_color,
                                       ascii_only=ascii_only, color=sys.stdout.isatty()))
            if board.is_terminal():
                outcome = board.outcome()
                print(f"Result: {outcome.result()} ({outcome.termination.name})")
                return 0
            if board.turn == human_color:
                move = human.act(board.clone())
            else:
                started = perf_counter()
                try:
                    move = bot.act(board.clone())
                    if not board.is_legal(move):
                        raise ValueError(f"Illegal action: {move!r}")
                    san = board.san(move)
                    board.step(move)
                except Exception as exc:
                    print(f"AI error ({bot.name}): {type(exc).__name__}: {exc}", file=sys.stderr)
                    return 1
                print(f"{bot.name}: {san} ({perf_counter()-started:.3f}s)")
                continue
            board.step(move)
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Chess Multi-Stage AI - Thang / Core & Stages 1-2")
    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--agent", choices=AGENT_REGISTRY, default="alphabeta")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--side", choices=("white","black"), default="white")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fen", default=None)
    parser.add_argument("--ascii", action="store_true", help="ASCII pieces for terminals without chess glyphs")
    args = parser.parse_args(argv)
    if args.depth < 1:
        parser.error("--depth must be >=1")
    try:
        initial_board = ChessBoard(args.fen)
    except ValueError as exc:
        parser.error(f"Invalid --fen: {exc}")
    if args.cli:
        return run_cli(args)
    from games.chess.gui import ChessGUI
    ChessGUI(args.agent, args.depth, args.side == "white", args.seed, initial_board).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
