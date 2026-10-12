"""Color-balanced matches; illegal actions/errors forfeit, ply caps are adjudicated draws."""
from dataclasses import dataclass, asdict, field
import chess
from time import perf_counter
from games.chess.board import ChessBoard

@dataclass
class GameResult:
    white: str
    black: str
    result: str
    reason: str
    moves: list
    initial_fen: str
    final_fen: str
    white_seconds: float
    black_seconds: float
    white_calls: int
    black_calls: int
    error: str = ""
    root_fen: str = ""
    initial_moves: list = field(default_factory=list)
    claim_draw: bool = True

    def to_dict(self):
        return asdict(self)

def restore_initial_board(record):
    """Restore history and draw policy from a GameResult dictionary."""
    state = ChessBoard(record.get("root_fen") or record["initial_fen"],
                       claim_draw=record.get("claim_draw", True))
    for uci in record.get("initial_moves", []):
        move = chess.Move.from_uci(uci)
        if not state.board.is_legal(move):
            raise ValueError(f"Illegal history move: {uci}")
        state.board.push(move)
    if state.fen() != record["initial_fen"]:
        raise ValueError("Initial FEN and history disagree")
    return state


def play_game(white, black, *, max_plies=300, board=None):
    if max_plies < 1:
        raise ValueError("max_plies must be positive")
    state = board.clone() if board is not None else ChessBoard()
    initial_fen = state.fen()
    times, calls, moves = [0.0, 0.0], [0, 0], []
    root_fen = state.board.root().fen(en_passant="fen")
    initial_moves = [m.uci() for m in state.board.move_stack]
    failures = {}
    for index, agent in enumerate((white, black)):
        try:
            agent.reset()
        except Exception as exc:
            failures[index] = f"{agent.name}: {type(exc).__name__}: {exc}"
    result, reason, error = "*", "", ""
    if failures:
        result = "1/2-1/2" if len(failures) == 2 else ("0-1" if 0 in failures else "1-0")
        reason = "double_forfeit" if len(failures) == 2 else "reset_forfeit"
        error = "; ".join(failures.values())
    while result == "*" and not state.is_terminal() and len(moves) < max_plies:
        index = 0 if state.turn else 1
        agent = (white, black)[index]
        started = perf_counter()
        try:
            move = agent.act(state.clone())
            if not state.is_legal(move):
                raise ValueError(f"Illegal action: {move!r}")
        except Exception as exc:
            result = "0-1" if index == 0 else "1-0"
            reason, error = "forfeit", f"{type(exc).__name__}: {exc}"
            break
        finally:
            times[index] += perf_counter() - started
            calls[index] += 1
        moves.append(move.uci())
        state.step(move)
    if result == "*":
        outcome = state.outcome()
        if outcome:
            result, reason = outcome.result(), outcome.termination.name.lower()
        else:
            result, reason = "1/2-1/2", "max_plies_adjudication"
    return GameResult(white.name, black.name, result, reason, moves, initial_fen,
                      state.fen(), *times, *calls, error, root_fen, initial_moves, state.claim_draw)

def evaluate(agent_a, agent_b, *, games=2, max_plies=300):
    if not isinstance(games, int) or games < 2 or games % 2:
        raise ValueError("games must be an even integer >=2 for balanced colors")
    records, wins, draws, losses = [], 0, 0, 0
    elapsed, calls = [0.0, 0.0], [0, 0]
    for i in range(games):
        swapped = i % 2 == 1
        record = play_game(agent_b if swapped else agent_a, agent_a if swapped else agent_b, max_plies=max_plies)
        records.append(record.to_dict())
        score = 0.5 if record.result == "1/2-1/2" else (1.0 if record.result == "1-0" else 0.0)
        if swapped:
            score = 1 - score
        wins += score == 1
        draws += score == 0.5
        losses += score == 0
        seconds = [record.white_seconds, record.black_seconds]
        counts = [record.white_calls, record.black_calls]
        for j in range(2):
            source = 1-j if swapped else j
            elapsed[j] += seconds[source]
            calls[j] += counts[source]
    return {"games": games, "wins": wins, "draws": draws, "losses": losses,
            "win_rate": wins/games, "draw_rate": draws/games,
            "score_rate": (wins + draws/2)/games,
            "a_ms_per_move": 1000*elapsed[0]/calls[0] if calls[0] else 0,
            "b_ms_per_move": 1000*elapsed[1]/calls[1] if calls[1] else 0,
            "records": records}
