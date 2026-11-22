"""Shared Stage 5 training summaries and fixed-opening validation helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time
from collections import Counter

import chess

from agents.stage4_supervised.supervised_agent import SupervisedAgent
from agents.stage5_rl.rl_agent import RLPolicyAgent
from arena.evaluator import play_game
from games.chess.board import ChessBoard


STAGE4 = Path("checkpoints/stage4/best_lichess.pt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_training(history: list[dict]) -> dict:
    return {
        "iterations": len(history),
        "games": sum(int(row["games"]) for row in history),
        "wins": sum(int(row["wins"]) for row in history),
        "draws": sum(int(row["draws"]) for row in history),
        "losses": sum(int(row["losses"]) for row in history),
        "truncated": sum(int(row["truncated"]) for row in history),
        "policy_games_used": sum(
            int(row["policy_games_used"]) for row in history
        ),
        "policy_games_skipped": sum(
            int(row["policy_games_skipped"]) for row in history
        ),
        "decisions": sum(int(row["decisions"]) for row in history),
        "policy_updates": sum(row["policy_loss"] is not None for row in history),
        "value_samples_used": sum(
            int(row["value_samples_used"]) for row in history
        ),
        "history": history,
    }


def reconstruct_opening(opening: dict) -> ChessBoard:
    board = ChessBoard(claim_draw=True)
    for uci in opening["moves_uci"]:
        move = chess.Move.from_uci(uci)
        if not board.board.is_legal(move):
            raise ValueError(f"Illegal saved opening move: {uci}")
        board.step(move)
    if board.fen() != opening["fen_after_8_plies"]:
        raise ValueError("Saved opening FEN differs from reconstructed history")
    if board.is_terminal():
        raise ValueError("Saved validation opening is terminal")
    return board


def evaluate_branch(branch: str, checkpoint: Path, openings: list[dict]) -> dict:
    learner = RLPolicyAgent(
        checkpoint, device="cpu", name=f"Stage5_{branch}"
    )
    stage4 = SupervisedAgent(STAGE4, device="cpu", name="Stage4_Elite")
    records = []
    wins = draws = losses = 0
    reasons = Counter()
    learner_seconds = stage4_seconds = 0.0
    learner_calls = stage4_calls = 0
    pair_scores = []
    started = time.perf_counter()
    for opening_index, opening in enumerate(openings, 1):
        board = reconstruct_opening(opening)
        pair_score = 0.0
        for learner_white in (True, False):
            white, black = (
                (learner, stage4) if learner_white else (stage4, learner)
            )
            result = play_game(white, black, max_plies=300, board=board)
            record = result.to_dict()
            record["opening_index"] = opening_index
            record["stage5_color"] = "white" if learner_white else "black"
            records.append(record)
            reasons[result.reason] += 1
            if result.result == "1/2-1/2":
                draws += 1
                score = 0.5
            else:
                learner_won = (
                    (learner_white and result.result == "1-0")
                    or (not learner_white and result.result == "0-1")
                )
                if learner_won:
                    wins += 1
                    score = 1.0
                else:
                    losses += 1
                    score = 0.0
            pair_score += score
            if learner_white:
                learner_seconds += result.white_seconds
                learner_calls += result.white_calls
                stage4_seconds += result.black_seconds
                stage4_calls += result.black_calls
            else:
                learner_seconds += result.black_seconds
                learner_calls += result.black_calls
                stage4_seconds += result.white_seconds
                stage4_calls += result.white_calls
        pair_scores.append(pair_score)
        if opening_index % 10 == 0:
            print(
                f"EVAL {branch} openings={opening_index}/50 "
                f"W/D/L={wins}/{draws}/{losses}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    errors = sum(bool(record["error"]) for record in records)
    return {
        "games": len(records),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": (wins + 0.5 * draws) / len(records),
        "termination_reasons": dict(reasons),
        "truncated": reasons.get("max_plies_adjudication", 0),
        "forfeits_or_errors": errors,
        "stage5_ms_per_move": (
            1000 * learner_seconds / learner_calls if learner_calls else 0.0
        ),
        "stage4_ms_per_move": (
            1000 * stage4_seconds / stage4_calls if stage4_calls else 0.0
        ),
        "elapsed_seconds": elapsed,
        "pair_scores": pair_scores,
        "records": records,
    }
