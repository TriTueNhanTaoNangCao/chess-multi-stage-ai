"""Color-balanced matches; illegal actions/errors forfeit, ply caps are adjudicated draws."""
from dataclasses import dataclass, asdict, field
import chess
from games.chess.board import ChessBoard
from arena.session import GameSession, measure_move

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
    telemetry: list = field(default_factory=list)

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
    session = GameSession(white.name, black.name, board=board, max_plies=max_plies)
    session.reset_agents((white, black))
    while not session.finished:
        session.accept(*measure_move((white, black)[session.index], session.state))
    return GameResult(**session.record())

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
