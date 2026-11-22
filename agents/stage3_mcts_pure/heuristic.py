"""
Hàm đánh giá thế cờ (Heuristic Position Evaluation) cho MCTS Pure tại điểm cắt rollout (cutoff).
Người thực hiện: Thi.

Quy ước:
- Đơn vị tính: Pawn units (Tốt = 1.0).
- Các trọng số vị trí, an toàn, cơ động là các giá trị nhỏ (0.01 - 0.50 pawn),
  đảm bảo thành phần vật chất luôn là yếu tố chi phối cốt lõi.
- Tính toán hoàn toàn đối xứng giữa Trắng và Đen:
    total_score = eval_side(board, WHITE) - eval_side(board, BLACK)
- Chuẩn hóa về [-1.0, 1.0] qua hàm:
    normalized = total_score / (|total_score| + 10.0)
  đảm bảo giá trị luôn hữu hạn, thuộc (-1.0, 1.0).
  Các kết quả ván cờ terminal (+1.0 Trắng thắng, -1.0 Đen thắng, 0.0 Hòa) được
  xử lý trước và không bao giờ bị ghi đè.
"""
from __future__ import annotations
import chess
from typing import Dict, List, Tuple

# 1. Bảng giá trị vật chất cơ sở (theo yêu cầu đề bài: P=1, N=3, B=3, R=5, Q=9)
PIECE_VALUES: Dict[chess.PieceType, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}

# 2. Bảng Piece-Square Tables (PST) theo Tomasz Michniewski (đơn vị: centipawns, nhân 0.001 -> pawn units)
# Bảng lưu theo thứ tự rank 8 xuống rank 1; index 0 là a8, index 63 là h1.
PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_TABLE = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_TABLE = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_TABLE = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]

# Vua ở Trung cuộc (ưu tiên nhập thành, ở góc an toàn)
KING_MG_TABLE = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]

# Vua ở Tàn cuộc (ưu tiên hoạt động, tiến vào trung tâm)
KING_EG_TABLE = [
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
]

PST_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
}

# Các ô trung tâm quan trọng
CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]


def compute_game_phase(board: chess.Board) -> float:
    """
    Tính giai đoạn ván cờ (game phase) chuẩn hóa trong [0.0, 1.0]:
    - 1.0: Khai cuộc toàn phần (đầy đủ quân).
    - 0.0: Tàn cuộc thuần túy (không còn quân nhẹ/nặng).
    Dựa trên trọng số phi-tốt: Knight: 1, Bishop: 1, Rook: 2, Queen: 4.
    Tổng tối đa cả 2 bên = 24.
    """
    weight = (
        len(board.pieces(chess.KNIGHT, chess.WHITE)) * 1 +
        len(board.pieces(chess.KNIGHT, chess.BLACK)) * 1 +
        len(board.pieces(chess.BISHOP, chess.WHITE)) * 1 +
        len(board.pieces(chess.BISHOP, chess.BLACK)) * 1 +
        len(board.pieces(chess.ROOK, chess.WHITE)) * 2 +
        len(board.pieces(chess.ROOK, chess.BLACK)) * 2 +
        len(board.pieces(chess.QUEEN, chess.WHITE)) * 4 +
        len(board.pieces(chess.QUEEN, chess.BLACK)) * 4
    )
    return min(1.0, max(0.0, weight / 24.0))


def evaluate_material(board: chess.Board, color: chess.Color) -> float:
    """
    Tính tổng điểm vật chất của một bên (đơn vị: pawn).
    P=1, N=3, B=3, R=5, Q=9. Vua không tính điểm vật chất.
    """
    return sum(
        len(board.pieces(pt, color)) * val
        for pt, val in PIECE_VALUES.items()
    )


def evaluate_piece_square_and_center(
    board: chess.Board,
    color: chess.Color,
    phase: float,
) -> float:
    """
    Đánh giá vị trí quân theo Piece-Square Tables và kiểm soát 4 ô trung tâm (d4, e4, d5, e5).
    Đơn vị: pawn units.
    - PST cho quân thường: tỉ lệ 0.001 pawn/cp (tối đa ±0.05 pawn/quân).
    - PST cho Vua: nội suy giữa King MG và King EG theo game phase.
    - Thưởng kiểm soát ô trung tâm: +0.03 pawn cho mỗi ô trung tâm bị kiểm soát hoặc có quân chiếm giữ.
    - Tránh thưởng quân bị treo/bắt: nếu quân bị đối phương tấn công mà không có quân mình bảo vệ,
      bị trừ 0.06 pawn.
    """
    score = 0.0
    opponent = not color

    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        table = PST_TABLES[pt]
        for sq in board.pieces(pt, color):
            # Với Trắng: sq 0 (a1) -> rank 1 -> index 56..63 qua square_mirror
            # Với Đen: sq 56 (a8) -> rank 8 -> index 56..63 đối xứng
            index = chess.square_mirror(sq) if color == chess.WHITE else sq
            pst_val = table[index] * 0.001  # centipawn -> pawn
            score += pst_val

            # Phạt nhẹ nếu quân đang bị tấn công mà không được bảo vệ (quân treo)
            if board.is_attacked_by(opponent, sq) and not board.is_attacked_by(color, sq):
                score -= 0.06

    # Đánh giá vị trí Vua theo phase
    king_sq = board.king(color)
    if king_sq is not None:
        index = chess.square_mirror(king_sq) if color == chess.WHITE else king_sq
        king_mg = KING_MG_TABLE[index] * 0.001
        king_eg = KING_EG_TABLE[index] * 0.001
        king_pst = phase * king_mg + (1.0 - phase) * king_eg
        score += king_pst

    # Kiểm soát trung tâm: d4, e4, d5, e5
    for c_sq in CENTER_SQUARES:
        if board.is_attacked_by(color, c_sq):
            score += 0.03
        piece = board.piece_at(c_sq)
        if piece is not None and piece.color == color:
            score += 0.03

    return score


def evaluate_development(
    board: chess.Board,
    color: chess.Color,
    phase: float,
) -> float:
    """
    Thưởng nhẹ phát triển Mã và Tượng rời khỏi ô xuất phát trong giai đoạn khai cuộc.
    Chỉ áp dụng khi phase >= 0.85 (khai cuộc, đủ quân).
    Không thưởng nếu quân bị đưa tới mép bàn (cột A hoặc H cho Mã) hoặc bị treo không có bảo vệ.
    Không cộng thưởng trong trung cuộc hoặc tàn cuộc.
    """
    if phase < 0.85:
        return 0.0

    score = 0.0
    opponent = not color

    if color == chess.WHITE:
        starting_knights = {chess.B1, chess.G1}
        starting_bishops = {chess.C1, chess.F1}
    else:
        starting_knights = {chess.B8, chess.G8}
        starting_bishops = {chess.C8, chess.F8}

    # Phát triển Mã
    for sq in board.pieces(chess.KNIGHT, color):
        if sq not in starting_knights:
            file_idx = chess.square_file(sq)
            # Tránh thưởng nếu Mã ở mép bàn cờ (cột a=0 hoặc h=7)
            if file_idx in (0, 7):
                continue
            # Tránh thưởng nếu quân bị tấn công không được bảo vệ
            if board.is_attacked_by(opponent, sq) and not board.is_attacked_by(color, sq):
                continue
            score += 0.08

    # Phát triển Tượng
    for sq in board.pieces(chess.BISHOP, color):
        if sq not in starting_bishops:
            # Tránh thưởng nếu quân bị tấn công không được bảo vệ
            if board.is_attacked_by(opponent, sq) and not board.is_attacked_by(color, sq):
                continue
            score += 0.08

    return score


def evaluate_pawns(
    board: chess.Board,
    color: chess.Color,
    phase: float,
) -> float:
    """
    Đánh giá cấu trúc Tốt (Pawns):
    - Thưởng tốt thông (passed pawn), tăng mạnh khi tiến gần hàng phong cấp và trong tàn cuộc.
    - Phạt tốt cô lập (isolated pawn - không có tốt cùng màu ở 2 cột liền kề).
    - Phạt tốt chồng (doubled pawn - nhiều hơn 1 tốt cùng màu trên 1 cột).
    """
    score = 0.0
    opponent = not color
    friendly_pawns = board.pieces(chess.PAWN, color)
    opponent_pawns = board.pieces(chess.PAWN, opponent)

    # Đếm số tốt trên từng cột để phát hiện tốt chồng và tốt cô lập
    file_counts = [0] * 8
    for sq in friendly_pawns:
        file_counts[chess.square_file(sq)] += 1

    for sq in friendly_pawns:
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        # Hàng tương đối (0 đến 7) nhìn từ phía người chơi
        rel_rank = rank_idx if color == chess.WHITE else (7 - rank_idx)

        # 1. Kiểm tra tốt thông (Passed pawn)
        # Không có tốt đối phương ở cột hiện tại và 2 cột bên cạnh ở phía trước
        is_passed = True
        adj_files = [f for f in (file_idx - 1, file_idx, file_idx + 1) if 0 <= f <= 7]
        for opp_sq in opponent_pawns:
            opp_file = chess.square_file(opp_sq)
            if opp_file in adj_files:
                opp_rank = chess.square_rank(opp_sq)
                opp_rel_rank = opp_rank if color == chess.WHITE else (7 - opp_rank)
                if opp_rel_rank > rel_rank:
                    is_passed = False
                    break

        if is_passed:
            # Tốt thông: thưởng tăng theo hàng tiến và tăng trong tàn cuộc
            base_passed = {
                1: 0.05,
                2: 0.08,
                3: 0.12,
                4: 0.20,
                5: 0.35,
                6: 0.60,
            }.get(rel_rank, 0.0)
            # Ở tàn cuộc, tốt thông có giá trị gấp 1.5 lần
            eg_multiplier = 1.0 + (1.0 - phase) * 0.5
            score += base_passed * eg_multiplier

        # 2. Tốt cô lập (Isolated pawn): không có tốt thân thiện nào ở cột lân cận
        left_has = file_counts[file_idx - 1] > 0 if file_idx > 0 else False
        right_has = file_counts[file_idx + 1] > 0 if file_idx < 7 else False
        if not left_has and not right_has:
            score -= 0.10

    # 3. Tốt chồng (Doubled pawn): phạt cho mỗi tốt dư thừa trên cùng một cột
    for count in file_counts:
        if count > 1:
            score -= 0.08 * (count - 1)

    return score


def evaluate_king_safety_and_activity(
    board: chess.Board,
    color: chess.Color,
    phase: float,
) -> float:
    """
    Đánh giá an toàn và hoạt động của Vua:
    - Ở Khai cuộc & Trung cuộc (phase lớn):
      + Thưởng lá chắn tốt quanh Vua (pawn shield): +0.04/tốt.
      + Phạt cột mở/nửa mở quanh Vua: -0.06/cột nửa mở, -0.10/cột mở hoàn toàn.
      + Phạt các ô lân cận Vua bị đối thủ tấn công (king ring attacks): -0.03/ô.
      + Phạt nhẹ nếu đang bị chiếu: -0.08 (chiếu hết đã được xử lý trước).
    - Ở Tàn cuộc (phase nhỏ):
      + Thưởng Vua hoạt động, tiến gần trung tâm: tối đa +0.15.
    """
    score = 0.0
    king_sq = board.king(color)
    if king_sq is None:
        return 0.0

    opponent = not color
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)

    # 1. An toàn Vua ở Khai cuộc / Trung cuộc (trọng số theo phase)
    if phase > 0.15:
        mg_weight = phase

        # Lá chắn tốt (Pawn shield) ngay trước mặt Vua
        shield_rank = king_rank + 1 if color == chess.WHITE else king_rank - 1
        shield_pawns = 0
        if 0 <= shield_rank <= 7:
            for f in (king_file - 1, king_file, king_file + 1):
                if 0 <= f <= 7:
                    check_sq = chess.square(f, shield_rank)
                    p = board.piece_at(check_sq)
                    if p is not None and p.piece_type == chess.PAWN and p.color == color:
                        shield_pawns += 1
        score += 0.04 * shield_pawns * mg_weight

        # Cột mở / nửa mở gần Vua
        for f in (king_file - 1, king_file, king_file + 1):
            if 0 <= f <= 7:
                friendly_on_file = any(
                    chess.square_file(sq) == f
                    for sq in board.pieces(chess.PAWN, color)
                )
                opponent_on_file = any(
                    chess.square_file(sq) == f
                    for sq in board.pieces(chess.PAWN, opponent)
                )
                if not friendly_on_file and not opponent_on_file:
                    score -= 0.10 * mg_weight  # Cột mở hoàn toàn
                elif not friendly_on_file:
                    score -= 0.06 * mg_weight  # Cột nửa mở

        # King ring: các ô lân cận Vua bị đối phương kiểm soát
        king_ring_attacks = 0
        for ring_sq in board.attacks(king_sq):
            if board.is_attacked_by(opponent, ring_sq):
                king_ring_attacks += 1
        score -= 0.03 * king_ring_attacks * mg_weight

        # Đang bị chiếu
        if board.is_check() and board.turn == color:
            score -= 0.08 * mg_weight

    # 2. Hoạt động của Vua ở Tàn cuộc (trọng số theo 1.0 - phase)
    if phase < 0.60:
        eg_weight = 1.0 - phase
        # Khoảng cách Chebyshev tới trung tâm (e4/d4/e5/d5)
        # Trung tâm có file 3..4, rank 3..4
        dist_file = min(abs(king_file - 3), abs(king_file - 4))
        dist_rank = min(abs(king_rank - 3), abs(king_rank - 4))
        dist_center = max(dist_file, dist_rank)  # 0 đến 3
        # Thưởng khi Vua ở gần trung tâm hơn
        score += 0.05 * (3 - dist_center) * eg_weight

    return score


def evaluate_mobility(
    board: chess.Board,
    color: chess.Color,
) -> float:
    """
    Đo lường khả năng cơ động (Mobility) nhanh và rẻ:
    - Không sinh nước đi hợp lệ toàn bộ (legal moves), không tạo bản sao bàn cờ.
    - Đếm tổng số ô mà các quân nhẹ và quân nặng (Mã, Tượng, Xe, Hậu) đang kiểm soát
      qua hàm bitboard C siêu tốc `board.attacks(sq)`.
    - Trọng số rất nhỏ: 0.003 pawn/ô kiểm soát để không làm biến dạng giá trị vật chất.
    """
    attack_count = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            attack_count += len(board.attacks(sq))
    return attack_count * 0.003


def evaluate_side(board: chess.Board, color: chess.Color, phase: float) -> float:
    """
    Tổng hợp toàn bộ các thành phần đánh giá cho một bên (color):
    1. Vật chất (Material) - thành phần chi phối chính
    2. Vị trí PST & kiểm soát trung tâm (PST & Center Control)
    3. Phát triển quân nhẹ ở khai cuộc (Minor Piece Development)
    4. Cấu trúc Tốt (Pawns: passed, isolated, doubled)
    5. An toàn và hoạt động Vua (King Safety & King Activity)
    6. Cơ động (Mobility)
    """
    mat = evaluate_material(board, color)
    pst_center = evaluate_piece_square_and_center(board, color, phase)
    dev = evaluate_development(board, color, phase)
    pawns = evaluate_pawns(board, color, phase)
    king = evaluate_king_safety_and_activity(board, color, phase)
    mob = evaluate_mobility(board, color)

    return mat + pst_center + dev + pawns + king + mob


def evaluate_position_raw(board: chess.Board) -> float:
    """
    Đánh giá thế cờ thô (raw score) theo góc nhìn quân Trắng:
        score = eval_side(board, WHITE) - eval_side(board, BLACK)
    Tính toán hoàn toàn đối xứng: lật màu bàn cờ (board.mirror()) sẽ đảo ngược dấu chính xác.
    """
    phase = compute_game_phase(board)
    white_score = evaluate_side(board, chess.WHITE, phase)
    black_score = evaluate_side(board, chess.BLACK, phase)
    return white_score - black_score


def evaluate_cutoff_score(board: chess.Board) -> float:
    """
    Hàm đánh giá tại điểm cắt rollout (cutoff).
    - Chuẩn hóa raw_score về khoảng (-1.0, 1.0) theo công thức:
        normalized = raw_score / (|raw_score| + 10.0)
    - Tương thích 100% với PureMCTS baseline:
      Nếu chỉ có chênh lệch vật chất delta, công thức trở về đúng delta / (|delta| + 10.0).
    - Không bao giờ trả về đúng +1.0 hoặc -1.0, giữ nguyên không gian cho kết quả ván cờ terminal.
    """
    raw_score = evaluate_position_raw(board)
    return raw_score / (abs(raw_score) + 10.0)
