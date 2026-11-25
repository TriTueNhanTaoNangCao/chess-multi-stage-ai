"""Single GUI/terminal entry point."""
import argparse
import sys
import chess
from agents import AGENT_REGISTRY, CHECKPOINT_REQUIRED_AGENTS, build_agent, HumanAgent
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
    from time import perf_counter, sleep
    just_fix_windows_console()
    board = ChessBoard(args.fen)
    ascii_only = args.ascii or needs_ascii(sys.stdout)
    if getattr(args, "watch", False):
        white_bot = build_agent(args.agent, args.depth, args.seed, checkpoint=getattr(args, "checkpoint", None),
                                move_time_limit_seconds=getattr(args, "move_time_limit", None))
        try:
            black_bot = build_agent(args.opponent, args.depth, args.seed + 1,
                                    move_time_limit_seconds=getattr(args, "move_time_limit", None))
        except BaseException:
            close = getattr(white_bot, "close", None)
            if close:
                close()
            raise
        try:
            while True:
                print(ChessRenderer.render(board.board, flipped=False,
                                           ascii_only=ascii_only, color=sys.stdout.isatty()))
                if board.is_terminal():
                    outcome = board.outcome()
                    print(f"Result: {outcome.result()} ({outcome.termination.name})")
                    return 0
                current_bot = white_bot if board.turn == chess.WHITE else black_bot
                turn_str = "Trắng" if board.turn == chess.WHITE else "Đen"
                started = perf_counter()
                try:
                    prepare = getattr(current_bot, "prepare", None)
                    if prepare is not None:
                        prepare()
                        started = perf_counter()
                    move = current_bot.act(board.clone())
                    if not board.is_legal(move):
                        raise ValueError(f"Illegal action: {move!r}")
                    san = board.san(move)
                    board.step(move)
                except Exception as exc:
                    print(f"AI error ({current_bot.name}): {type(exc).__name__}: {exc}", file=sys.stderr)
                    return 1
                print(f"{turn_str} [{current_bot.name}]: {san} ({perf_counter()-started:.3f}s)")
                if getattr(args, "delay", 0) > 0:
                    sleep(args.delay)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        finally:
            for agent in (white_bot, black_bot):
                close = getattr(agent, "close", None)
                if close:
                    close()

    bot = build_agent(args.agent, args.depth, args.seed, checkpoint=getattr(args, "checkpoint", None),
                       move_time_limit_seconds=getattr(args, "move_time_limit", None))
    human = HumanAgent()
    human_color = args.side == "white"
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
                    prepare = getattr(bot, "prepare", None)
                    if prepare is not None:
                        prepare()
                        started = perf_counter()
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
    finally:
        close = getattr(bot, "close", None)
        if close:
            close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Chess Multi-Stage AI - Thang / Core & Stages 1-2")
    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--agent", choices=AGENT_REGISTRY, default="alphabeta")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--move-time-limit", type=float, default=None, metavar="SECONDS",
                        help="Optional maximum think time per AI move, e.g. 10 or 15")
    parser.add_argument("--side", choices=("white","black"), default="white")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fen", default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=("Path to model checkpoint (required for supervised/rl_policy; "
              "Stage 4 baseline: checkpoints/stage4/best_lichess.pt)"),
    )
    parser.add_argument("--ascii", action="store_true", help="ASCII pieces for terminals without chess glyphs")
    parser.add_argument("--watch", action="store_true", help="Watch bot vs bot in Pygame GUI without human player")
    parser.add_argument("--opponent", choices=AGENT_REGISTRY, default="random", help="Opponent bot in watch mode (default: random)")
    parser.add_argument("--delay", type=float, default=0.4, help="Delay between bot moves in seconds (default: 0.4)")
    args = parser.parse_args(argv)
    if args.depth < 1:
        parser.error("--depth must be >=1")
    from agents.search_limits import validate_seconds
    try:
        validate_seconds(args.move_time_limit)
    except ValueError as exc:
        parser.error(str(exc))
    if args.agent in CHECKPOINT_REQUIRED_AGENTS and not args.checkpoint:
        parser.error(f"--checkpoint is required when using {args.agent} agent")
    try:
        initial_board = ChessBoard(args.fen)
    except ValueError as exc:
        parser.error(f"Invalid --fen: {exc}")
    if args.cli:
        return run_cli(args)
    from games.chess.gui import ChessGUI
    if args.watch:
        ChessGUI(
            agent_key=args.agent,
            depth=args.depth,
            human_color=None,
            seed=args.seed,
            board=initial_board,
            black_agent_key=args.opponent,
            move_delay=args.delay,
            checkpoint=args.checkpoint,
            move_time_limit_seconds=args.move_time_limit,
        ).run()
        return 0
    ChessGUI(
        agent_key=args.agent,
        depth=args.depth,
        human_color=args.side == "white",
        seed=args.seed,
        board=initial_board,
        checkpoint=args.checkpoint,
        move_time_limit_seconds=args.move_time_limit,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
