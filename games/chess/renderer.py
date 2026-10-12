"""
Rich Unicode and ANSI terminal renderer for ChessBoard.
"""

import chess
from typing import Optional

# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Square backgrounds
BG_LIGHT = "\033[48;2;240;217;181m"  # Classic wood light
BG_DARK = "\033[48;2;181;136;99m"    # Classic wood dark
BG_SELECTED = "\033[48;2;186;202;68m" # Highlight green
BG_CHECK = "\033[48;2;235;97;80m"     # Red check alert

# Piece foreground colors
FG_WHITE = "\033[38;2;255;255;255m"
FG_BLACK = "\033[38;2;30;30;30m"
FG_COORD = "\033[38;2;160;160;160m"

# Unicode Chess Symbols
PIECE_SYMBOLS = {
    chess.Piece(chess.PAWN, chess.WHITE): "♙",
    chess.Piece(chess.KNIGHT, chess.WHITE): "♘",
    chess.Piece(chess.BISHOP, chess.WHITE): "♗",
    chess.Piece(chess.ROOK, chess.WHITE): "♖",
    chess.Piece(chess.QUEEN, chess.WHITE): "♕",
    chess.Piece(chess.KING, chess.WHITE): "♔",
    chess.Piece(chess.PAWN, chess.BLACK): "♟",
    chess.Piece(chess.KNIGHT, chess.BLACK): "♞",
    chess.Piece(chess.BISHOP, chess.BLACK): "♝",
    chess.Piece(chess.ROOK, chess.BLACK): "♜",
    chess.Piece(chess.QUEEN, chess.BLACK): "♛",
    chess.Piece(chess.KING, chess.BLACK): "♚",
}


class ChessRenderer:
    """Renders chess boards beautifully in the terminal with ANSI colors."""

    @staticmethod
    def render(
        board: chess.Board,
        last_move: Optional[chess.Move] = None,
        flipped: bool = False,
        ascii_only: bool = False,
        color: bool = True,
    ) -> str:
        """Render board as formatted ANSI string."""
        lines = []
        is_check = board.is_check()
        check_sq = board.king(board.turn) if is_check else None

        last_from = last_move.from_square if last_move else None
        last_to = last_move.to_square if last_move else None

        ranks = range(8) if flipped else range(7, -1, -1)
        files = range(7, -1, -1) if flipped else range(8)

        # Header
        turn_str = ("White" if board.turn else "Black") if ascii_only else ("White (♔)" if board.turn else "Black (♚)")
        status_info = f"{BOLD}Turn: {turn_str}{RESET}"
        if board.is_checkmate():
            status_info += f" | {BOLD}\033[31mCHECKMATE!{RESET}"
        elif board.is_stalemate():
            status_info += f" | {BOLD}\033[33mSTALEMATE (DRAW)!{RESET}"
        elif is_check:
            status_info += f" | {BOLD}\033[31mCHECK!{RESET}"
        lines.append(status_info)
        lines.append("")

        for r in ranks:
            rank_label = f"{FG_COORD}{r + 1} {RESET}"
            row_str = rank_label
            for c in files:
                sq = chess.square(c, r)
                piece = board.piece_at(sq)

                # Determine background color
                is_light_sq = (r + c) % 2 != 0
                bg = BG_LIGHT if is_light_sq else BG_DARK
                if sq == check_sq:
                    bg = BG_CHECK
                elif sq in (last_from, last_to):
                    bg = BG_SELECTED

                # Determine piece symbol & color
                if piece:
                    symbol = piece.symbol() if ascii_only else PIECE_SYMBOLS.get(piece, piece.symbol())
                    fg = FG_WHITE if piece.color == chess.WHITE else FG_BLACK
                    cell = f"{bg}{fg}{BOLD} {symbol} {RESET}"
                else:
                    cell = f"{bg}   {RESET}"

                row_str += cell
            lines.append(row_str)

        # File coordinate labels at bottom
        file_letters = "   " + "  ".join(
            [chess.FILE_NAMES[c].upper() for c in files]
        )
        lines.append(f"{FG_COORD}{file_letters}{RESET}")

        result = "\n".join(lines)
        if not color:
            import re
            result = re.sub(r"\x1b\[[0-9;]*m", "", result)
        return result

    @staticmethod
    def print_board(board: chess.Board, last_move: Optional[chess.Move] = None) -> None:
        """Directly print rendered board to console."""
        print(ChessRenderer.render(board, last_move))
