"""Kiểm thử Stage 7 v2: MCTS theo lô, nhãn Value, replay buffer, checkpoint, bước học."""
import chess
import numpy as np
import pytest
import torch

from agents.stage7_alphazero import AlphaZeroAgent
from agents.stage7_alphazero.batched_mcts import (
    SearchConfig, label_values, play_games, search_position, terminal_value,
)
from games.chess.board import ChessBoard
from games.chess.encoder import ChessEncoder, NUM_ACTIONS
from models.dual_head_resnet import DualHeadResNet
from training.stage7_train_alphazero_v2 import (
    ReplayBuffer, compute_loss, make_payload, parse_args, train_steps, build_optimizer,
)


class FakeNet:
    """Mạng giả: logit ngẫu nhiên, giá trị ngẫu nhiên nhỏ."""

    def __init__(self, seed=0):
        self.gen = torch.Generator().manual_seed(seed)

    def eval(self):
        return self

    def __call__(self, x):
        b = x.shape[0]
        return (torch.randn(b, NUM_ACTIONS, generator=self.gen) * 0.5,
                torch.tanh(torch.randn(b, 1, generator=self.gen) * 0.3))


def test_search_visits_equal_simulations():
    root = search_position(FakeNet(1), "cpu", chess.Board(), 40, seed=3)
    assert root.N.sum() == 40
    assert root.n == 40
    assert len(root.moves) == 20


def test_search_does_not_mutate_board():
    board = chess.Board()
    board.push_san("e4")
    fen, stack = board.fen(), len(board.move_stack)
    search_position(FakeNet(2), "cpu", board, 30)
    assert board.fen() == fen and len(board.move_stack) == stack


def test_terminal_value():
    mate = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    mate.push(chess.Move.from_uci("a1a8"))
    assert terminal_value(mate, list(mate.legal_moves)) == -1.0
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert terminal_value(stalemate, list(stalemate.legal_moves)) == 0.0
    assert terminal_value(chess.Board(), list(chess.Board().legal_moves)) is None


def test_label_values():
    assert label_values([True, False, True], True) == [1.0, -1.0, 1.0]
    assert label_values([True, False, True], False) == [-1.0, 1.0, -1.0]
    assert label_values([True, False], None) == [0.0, 0.0]


def test_selfplay_records_consistent():
    cfg = SearchConfig(simulations=20, parallel_games=4, max_plies=60)
    recs, stats = play_games(FakeNet(5), "cpu", 6, cfg, seed=11)
    assert len(recs) == 6
    start_idx = set(ChessEncoder.legal_action_indices(chess.Board()))
    for r in recs:
        t = len(r["z"])
        assert r["states"].shape == (t, 14, 8, 8) and r["states"].dtype == np.uint8
        assert len(r["idxs"]) == t == len(r["probs"]) == len(r["vmask"])
        for p, ix in zip(r["probs"], r["idxs"]):
            assert abs(float(p.sum()) - 1.0) < 1e-4 and len(p) == len(ix)
        assert set(r["idxs"][0].tolist()) == start_idx
        if r["result"] == "capped":
            assert not r["vmask"].any() and (r["z"] == 0).all() and r["plies"] == 60
        else:
            assert r["vmask"].all()
            if r["result"] in ("white", "black"):
                sign = 1.0 if r["result"] == "white" else -1.0
                expected = np.array([sign if i % 2 == 0 else -sign for i in range(t)], dtype=np.float32)
                assert (r["z"] == expected).all()
            else:
                assert (r["z"] == 0).all()
    assert stats["games"] == 6


def test_capped_games_have_no_value_label():
    cfg = SearchConfig(simulations=8, parallel_games=2, max_plies=6)
    recs, _ = play_games(FakeNet(7), "cpu", 2, cfg, seed=1)
    for r in recs:
        assert r["result"] == "capped"
        assert not r["vmask"].any()


def test_eval_random_runs():
    cfg = SearchConfig(simulations=8, parallel_games=4, max_plies=50)
    recs, _ = play_games(FakeNet(9), "cpu", 4, cfg, seed=2, eval_random=True)
    assert len(recs) == 4
    assert all(r["result"] in ("win", "draw", "loss", "capped") for r in recs)


def test_masked_value_loss():
    logits = torch.zeros(3, NUM_ACTIONS)
    values = torch.tensor([[0.9], [0.9], [0.9]])
    pi = torch.zeros(3, NUM_ACTIONS)
    pi[:, 0] = 1.0
    z = torch.zeros(3)
    _, _, v_all = compute_loss(logits, values, pi, z, torch.ones(3))
    _, _, v_none = compute_loss(logits, values, pi, z, torch.zeros(3))
    assert v_all.item() == pytest.approx(0.81, abs=1e-5)
    assert v_none.item() == 0.0


def _small_buffer(n_games=4, sims=10, cap=5000):
    cfg = SearchConfig(simulations=sims, parallel_games=n_games, max_plies=40)
    recs, _ = play_games(FakeNet(3), "cpu", n_games, cfg, seed=5)
    buf = ReplayBuffer(cap, seed=0)
    buf.add_records(recs)
    return buf


def test_replay_buffer_sample_and_roundtrip(tmp_path):
    buf = _small_buffer()
    x, pi, z, m = buf.sample(16, "cpu")
    assert x.shape == (16, 14, 8, 8) and pi.shape == (16, NUM_ACTIONS)
    assert torch.allclose(pi.sum(dim=1), torch.ones(16), atol=1e-4)
    path = tmp_path / "replay.npz"
    buf.save(path)
    buf2 = ReplayBuffer(5000, seed=0)
    buf2.load(path)
    assert len(buf2) == len(buf)
    assert (buf2.data[0][0] == buf.data[0][0]).all()
    assert (buf2.data[3][1] == buf.data[3][1]).all()


def test_replay_buffer_is_circular():
    buf = ReplayBuffer(50, seed=0)
    cfg = SearchConfig(simulations=5, parallel_games=2, max_plies=40)
    recs, _ = play_games(FakeNet(1), "cpu", 2, cfg, seed=1)
    buf.add_records(recs)
    buf.add_records(recs)
    assert len(buf) == 50


def test_checkpoint_loadable_by_agent(tmp_path):
    args = parse_args(["--smoke"])
    model = DualHeadResNet(hidden_dim=16, num_res_blocks=1)
    path = tmp_path / "best.pt"
    torch.save(make_payload(model, args, 3, {"loss": 1.0}), path)
    agent = AlphaZeroAgent(checkpoint=path, simulations=6, seed=1)
    board = ChessBoard()
    move = agent.act(board)
    assert move in board.get_legal_actions()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("training_config", "metrics", "state_dict", "model_config", "stage"):
        assert key in payload


def test_training_reduces_policy_loss():
    torch.manual_seed(0)
    buf = _small_buffer(n_games=6, sims=10)
    args = parse_args(["--smoke"])
    model = DualHeadResNet(hidden_dim=16, num_res_blocks=1)
    opt = build_optimizer(model, args)
    first = train_steps(model, opt, buf, 5, 32, "cpu", 10.0)
    for _ in range(8):
        last = train_steps(model, opt, buf, 20, 32, "cpu", 10.0)
    assert last["policy_loss"] < first["policy_loss"]


def test_material_adjudication_labels_capped_games():
    import chess
    from agents.stage7_alphazero.batched_mcts import adjudicated_white_value, material_balance

    b = chess.Board()
    assert material_balance(b) == 0
    assert adjudicated_white_value(b, 3.0) is None
    b.remove_piece_at(chess.D8)  # Đen mất Hậu: Trắng hơn 9 điểm
    assert material_balance(b) == 9
    assert adjudicated_white_value(b, 3.0) == 1.0
    b2 = chess.Board()
    b2.remove_piece_at(chess.A7)  # Đen mất 1 tốt: chưa đủ ngưỡng 3
    assert adjudicated_white_value(b2, 3.0) is None
    b3 = chess.Board()
    b3.remove_piece_at(chess.B1)  # Trắng mất Mã: Đen hơn 3 điểm, giá trị -0.5 theo góc nhìn Trắng
    assert adjudicated_white_value(b3, 3.0) == -0.5
    assert adjudicated_white_value(b, 0.0) is None  # tắt


def test_capped_games_with_adjudication_have_value_labels_and_flag():
    cfg = SearchConfig(simulations=6, parallel_games=4, max_plies=20, adjudicate_material=1.0)
    recs, _ = play_games(FakeNet(3), "cpu", 8, cfg, seed=3)
    for r in recs:
        if r["result"] == "capped":
            assert "adjudicated" in r
            if r["adjudicated"]:
                assert r["vmask"].all() and np.abs(r["z"]).max() > 0
            else:
                assert not r["vmask"].any()


def test_random_opening_changes_start_and_not_recorded():
    cfg = SearchConfig(simulations=4, parallel_games=4, max_plies=6, random_opening_plies=6)
    recs, _ = play_games(FakeNet(5), "cpu", 8, cfg, seed=5)
    assert len(recs) == 8
    assert all(r["plies"] <= 6 for r in recs)  # ply đếm từ lúc mạng bắt đầu chơi, không tính nước ngẫu nhiên
    firsts = {r["states"][0].tobytes() for r in recs if len(r["states"])}
    assert len(firsts) > 1  # các ván bắt đầu từ các thế khác nhau


def _fake_finished_game(remove_square=None):
    from agents.stage7_alphazero.batched_mcts import _Game

    g = _Game(0, "selfplay", None, True, True)
    if remove_square is not None:
        g.board.remove_piece_at(remove_square)  # Đen mất quân, Trắng hơn vật chất
    st = np.zeros((14, 8, 8), np.uint8)
    idx = np.array([0, 1], np.int16)
    pr = np.array([0.5, 0.5], np.float32)
    g.samples = [(st, idx, pr, chess.WHITE), (st, idx, pr, chess.BLACK)]
    g.ply = 2
    return g


def test_adjudicate_draws_labels_repetition_draw_by_material():
    from agents.stage7_alphazero.batched_mcts import _finish

    rep = chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
    cfg = SearchConfig(adjudicate_material=3.0, adjudicate_draws=True)
    rec = _finish(_fake_finished_game(chess.D8), rep, cfg)  # Trắng hơn Hậu (9)
    assert rec["result"] == "draw" and rec["adjudicated"] is True
    assert rec["z"].tolist() == [1.0, -1.0]  # theo góc nhìn bên đi ở từng mẫu
    # tắt cờ: giữ nguyên nhãn hòa 0
    rec_off = _finish(_fake_finished_game(chess.D8), rep, SearchConfig(adjudicate_material=3.0))
    assert rec_off["adjudicated"] is False and rec_off["z"].tolist() == [0.0, 0.0]


def test_adjudicate_draws_skips_insufficient_material_and_stalemate():
    from agents.stage7_alphazero.batched_mcts import _finish

    cfg = SearchConfig(adjudicate_material=3.0, adjudicate_draws=True)
    for term in (chess.Termination.INSUFFICIENT_MATERIAL, chess.Termination.STALEMATE):
        rec = _finish(_fake_finished_game(chess.D8), chess.Outcome(term, None), cfg)
        assert rec["adjudicated"] is False and rec["z"].tolist() == [0.0, 0.0]
    # ván có người thắng thật không bị đổi nhãn
    win = chess.Outcome(chess.Termination.CHECKMATE, chess.BLACK)
    rec = _finish(_fake_finished_game(chess.D8), win, cfg)
    assert rec["z"].tolist() == [-1.0, 1.0] and rec["adjudicated"] is False
