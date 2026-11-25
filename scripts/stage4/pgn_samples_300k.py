"""Sample at most 32 distinct pre-move positions from one Stage 4 candidate.

No splitting, dataset IO or fingerprint acceptance occurs here. The caller may
accept the candidate's fingerprint only after this function returns samples and
any remaining downstream checks succeed.
"""
from dataclasses import dataclass

import chess.pgn
import numpy as np

from games.chess.encoder import ChessEncoder, NUM_ACTIONS


@dataclass(frozen=True)
class PositionSample:
    state: np.ndarray       # float32 HWC (8, 8, 14), before the recorded move.
    target: int             # Existing 4672-action mapping, oriented by board.turn.
    legal_mask: np.ndarray  # bool (4672,), claim_draw=False.
    ply_index: int          # Zero-based index in the PGN mainline.


def extract_samples(
    game: chess.pgn.Game,
    rng: np.random.Generator,
) -> list[PositionSample] | None:
    """Return samples in ply order, or None to reject the entire game.

Use the caller's RNG, uniformly sampling min(32, number_of_plies) indices
without replacement. Replay every move on a private board to retain the full
history. Claims of draws do not stop PGN replay, but automatic terminal states
do. Partial samples are discarded on any invalid move, terminal pre-move state,
encoding error or target/mask mismatch. No global RNG state is used.
"""
    if game.errors:
        return None
    try:
        board = game.board()
        moves = tuple(game.mainline_moves())
    except (ValueError, IndexError):
        return None
    if not board.is_valid() or not moves:
        return None

    selected = set(int(index) for index in rng.choice(
        len(moves), size=min(32, len(moves)), replace=False,
    ))
    samples: list[PositionSample] = []
    for ply_index, move in enumerate(moves):
        if board.is_game_over(claim_draw=False) or not board.is_legal(move):
            return None
        if ply_index in selected:
            try:
                target = ChessEncoder.encode_move(move, board.turn)
                mask = ChessEncoder.get_legal_action_mask(
                    board, claim_draw=False,
                ).astype(np.bool_, copy=False)
                if not (0 <= target < NUM_ACTIONS) or not mask[target]:
                    return None
                if ChessEncoder.decode_move(target, board, claim_draw=False) != move:
                    return None
                state = ChessEncoder.encode_board(board)
            except (ValueError, IndexError):
                return None
            samples.append(PositionSample(state, target, mask, ply_index))
        board.push(move)
    return samples
