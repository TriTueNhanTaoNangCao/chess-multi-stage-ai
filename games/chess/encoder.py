"""Project 14-plane features and AlphaZero-style 4672 action mapping."""
from typing import Dict, List, Optional, Tuple
import chess
import numpy as np
from games.chess.board import ChessBoard


def _raw(board):
    if isinstance(board, ChessBoard):
        return board.board
    if isinstance(board, chess.Board):
        return board
    raise TypeError("Expected ChessBoard or chess.Board")


def _claim_policy(board, claim_draw):
    if claim_draw is not None:
        return claim_draw
    return board.claim_draw if isinstance(board, ChessBoard) else True


NUM_ACTIONS = 4672  # 8 * 8 * 73
NUM_CHANNELS = 14

# Queen move directions: (delta_rank, delta_file)
# N, NE, E, SE, S, SW, W, NW (relative to current player looking forward)
QUEEN_DIRECTIONS = [
    (1, 0),   # N
    (1, 1),   # NE
    (0, 1),   # E
    (-1, 1),  # SE
    (-1, 0),  # S
    (-1, -1), # SW
    (0, -1),  # W
    (1, -1),  # NW
]

# Knight move offsets: (delta_rank, delta_file)
KNIGHT_OFFSETS = [
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
]

# Underpromotion directions: (delta_file) - left capture (-1), straight (0), right capture (+1)
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _build_action_mappings():
    """Build precalculated dictionaries mapping between (from_sq, move_type) and action_idx."""
    move_to_idx: Dict[Tuple[int, int, Optional[int]], int] = {}
    idx_to_move_spec: Dict[int, Tuple[int, int, Optional[int]]] = {}

    for from_sq in range(64):
        from_rank = from_sq // 8
        from_file = from_sq % 8

        plane_idx = 0

        # 1. 56 Queen moves
        for d_idx, (dr, df) in enumerate(QUEEN_DIRECTIONS):
            for dist in range(1, 8):
                to_rank = from_rank + dr * dist
                to_file = from_file + df * dist
                action_idx = from_sq * 73 + plane_idx
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_sq = to_rank * 8 + to_file
                    move_to_idx[(from_sq, to_sq, None)] = action_idx
                    idx_to_move_spec[action_idx] = (from_sq, to_sq, None)
                plane_idx += 1

        # 2. 8 Knight moves
        for dr, df in KNIGHT_OFFSETS:
            to_rank = from_rank + dr
            to_file = from_file + df
            action_idx = from_sq * 73 + plane_idx
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_sq = to_rank * 8 + to_file
                move_to_idx[(from_sq, to_sq, None)] = action_idx
                idx_to_move_spec[action_idx] = (from_sq, to_sq, None)
            plane_idx += 1

        # 3. 9 Underpromotions (from rank 6 to rank 7 only)
        for df in [-1, 0, 1]:
            for piece in UNDERPROMOTION_PIECES:
                action_idx = from_sq * 73 + plane_idx
                if from_rank == 6:
                    to_file = from_file + df
                    to_rank = 7
                    if 0 <= to_file < 8:
                        to_sq = to_rank * 8 + to_file
                        move_to_idx[(from_sq, to_sq, piece)] = action_idx
                        idx_to_move_spec[action_idx] = (from_sq, to_sq, piece)
                plane_idx += 1

    return move_to_idx, idx_to_move_spec


MOVE_TO_IDX, IDX_TO_MOVE_SPEC = _build_action_mappings()


class ChessEncoder:
    """Accepts raw/wrapped boards; action APIs respect the draw-claim policy."""

    @staticmethod
    def encode_board(board: chess.Board | ChessBoard) -> np.ndarray:
        """HWC (8,8,14): White P..K, Black P..K, turn, en-passant.

        Rank axis 0 = rank 1. Tensor alone is NOT a complete rules state;
        use encode_state/decode_state to preserve rights, clocks and history.
        """
        board = _raw(board)
        planes = np.zeros((8, 8, NUM_CHANNELS), dtype=np.float32)
        for sq, piece in board.piece_map().items():
            channel = piece.piece_type - 1 + (0 if piece.color else 6)
            planes[chess.square_rank(sq), chess.square_file(sq), channel] = 1
        planes[:, :, 12] = float(board.turn)
        if board.ep_square is not None:
            planes[chess.square_rank(board.ep_square), chess.square_file(board.ep_square), 13] = 1
        return planes

    @staticmethod
    def encode_board_tensor(board, device=None):
        """Lazy PyTorch conversion to NCHW (1,14,8,8)."""
        import torch
        tensor = torch.from_numpy(ChessEncoder.encode_board(board).transpose(2, 0, 1).copy()).unsqueeze(0)
        return tensor.to(device) if device is not None else tensor

    @staticmethod
    def encode_state(board):
        raw = _raw(board)
        return {"planes": ChessEncoder.encode_board(raw),
                "root_fen": raw.root().fen(en_passant="fen"),
                "initial_fen": board._initial_fen if isinstance(board, ChessBoard) else raw.root().fen(en_passant="fen"),
                "claim_draw": _claim_policy(board, None),
                "moves": [m.uci() for m in raw.move_stack]}

    @staticmethod
    def decode_state(state, *, as_wrapper=False):
        board = chess.Board(state["root_fen"])
        if not board.is_valid():
            raise ValueError("Invalid root position")
        for move in state["moves"]:
            board.push_uci(move)
        if not np.array_equal(ChessEncoder.encode_board(board), state["planes"]):
            raise ValueError("Tensor and metadata disagree")
        if as_wrapper:
            wrapper = ChessBoard(state.get("initial_fen", state["root_fen"]),
                                 claim_draw=state.get("claim_draw", True))
            wrapper.board = board
            return wrapper
        return board

    @staticmethod
    def orient_square(sq: int, turn: chess.Color) -> int:
        """Flip square vertically if Black is to move."""
        if turn == chess.WHITE:
            return sq
        r = 7 - (sq // 8)
        c = sq % 8
        return r * 8 + c

    @staticmethod
    def encode_move(move: chess.Move, turn: chess.Color) -> int:
        """
        Convert chess.Move to AlphaZero action index in [0..4671].
        Accounts for player orientation.
        """
        from_sq = ChessEncoder.orient_square(move.from_square, turn)
        to_sq = ChessEncoder.orient_square(move.to_square, turn)

        promo = move.promotion
        # Queen promotions count as normal queen moves
        if promo == chess.QUEEN:
            promo = None

        key = (from_sq, to_sq, promo)
        if key in MOVE_TO_IDX:
            return MOVE_TO_IDX[key]

        raise ValueError(f"Unable to encode move: {move.uci()} (key={key})")

    @staticmethod
    def decode_move(action_idx: int, board, *, claim_draw=None) -> Optional[chess.Move]:
        """
        Decode action index back to legal chess.Move given current board.
        Returns None if decoded move is invalid or illegal.
        """
        policy = _claim_policy(board, claim_draw)
        board = _raw(board)
        if board.is_game_over(claim_draw=policy):
            return None
        if action_idx not in IDX_TO_MOVE_SPEC:
            return None

        from_sq_oriented, to_sq_oriented, promo = IDX_TO_MOVE_SPEC[action_idx]
        turn = board.turn

        from_sq = ChessEncoder.orient_square(from_sq_oriented, turn)
        to_sq = ChessEncoder.orient_square(to_sq_oriented, turn)

        # Check if this is a pawn promotion to Queen
        piece = board.piece_at(from_sq)
        if (
            promo is None
            and piece is not None
            and piece.piece_type == chess.PAWN
            and (to_sq // 8 == 7 or to_sq // 8 == 0)
        ):
            candidate = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
            if candidate in board.legal_moves:
                return candidate

        candidate = chess.Move(from_sq, to_sq, promotion=promo)
        if candidate in board.legal_moves:
            return candidate

        return None

    @staticmethod
    def get_legal_action_mask(board, *, claim_draw=None) -> np.ndarray:
        """Return boolean/binary mask of shape (4672,) indicating legal actions."""
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for idx in ChessEncoder.legal_action_indices(board, claim_draw=claim_draw):
            mask[idx] = 1.0
        return mask

    @staticmethod
    def legal_action_indices(board, *, claim_draw=None) -> List[int]:
        """List of valid action indices for current board position."""
        policy = _claim_policy(board, claim_draw)
        board = _raw(board)
        if board.is_game_over(claim_draw=policy):
            return []
        turn = board.turn
        indices = []
        for move in board.legal_moves:
            indices.append(ChessEncoder.encode_move(move, turn))
        return indices
