"""
Stage 6: AlphaGo Agent kết hợp Policy Prior (Stage 4), Value Network (Stage 5) và MCTS.
Người thực hiện: Tùng.

Tuân thủ nghiêm ngặt hợp đồng kỹ thuật (docs/HANDOFF.md):
- Nhận ChessBoard, không bao giờ làm biến đổi bàn cờ của bên gọi.
- Ném ValueError nếu vị trí đã kết thúc (terminal position).
- Trả về một chess.Move hợp lệ (legal move).
- Hỗ trợ nạp checkpoint Policy và Value chuẩn schema HANDOFF.md.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import chess
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from models.cnn_policy import CNNPolicyNetwork
    from models.value_net import ValueNetwork
except ModuleNotFoundError as err:
    if err.name == "torch" or (err.name and err.name.startswith("torch.")):
        torch = None
        CNNPolicyNetwork = None
        ValueNetwork = None
    else:
        raise

from agents.base import BaseAgent
from games.chess.board import ChessBoard
from games.chess.encoder import ChessEncoder, NUM_ACTIONS
from .mcts import AlphaGoMCTS


class AlphaGoAgent(BaseAgent):
    """
    Agent cờ vua thế hệ thứ 6: AlphaGo-style MCTS.
    Sử dụng Policy Network để định hướng mở rộng nhánh (Prior probability P)
    và Value Network kết hợp Fast Rollout để đánh giá giá trị nút lá.
    """

    def __init__(
        self,
        checkpoint_policy: Optional[Union[str, Path]] = None,
        checkpoint_value: Optional[Union[str, Path]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        simulations: int = 100,
        c_puct: float = 1.5,
        rollout_weight: float = 0.5,
        max_rollout_depth: int = 15,
        device: str = "cpu",
        seed: Optional[int] = None,
        name: Optional[str] = None,
        player_id: Optional[int] = None,
        policy_net: Optional[Any] = None,
        value_net: Optional[Any] = None,
    ):
        name = name or f"AlphaGo ({simulations} sims)"
        super().__init__(name=name, player_id=player_id)

        self.device = torch.device(device) if torch is not None else None
        self.simulations = simulations
        self.c_puct = c_puct
        self.rollout_weight = rollout_weight
        self.max_rollout_depth = max_rollout_depth
        self.seed = seed

        if seed is not None:
            if torch is not None:
                torch.manual_seed(seed)
            np.random.seed(seed)

        # 1. Xử lý Policy Network
        self.policy_net = policy_net
        if self.policy_net is None and (checkpoint_policy or checkpoint):
            ckpt_to_check = checkpoint_policy or checkpoint
            self.policy_net = self._load_policy_model(ckpt_to_check)

        # Nếu không có checkpoint nhưng torch có sẵn và không chỉ định mạng, khởi tạo mặc định
        if self.policy_net is None and torch is not None and CNNPolicyNetwork is not None:
            self.policy_net = CNNPolicyNetwork()
            self.policy_net.to(self.device)
            self.policy_net.eval()

        # 2. Xử lý Value Network
        self.value_net = value_net
        if self.value_net is None and checkpoint_value:
            self.value_net = self._load_value_model(checkpoint_value)
        elif self.value_net is None and checkpoint:
            # Kiểm tra nếu cờ --checkpoint chung trỏ tới file ValueNetwork
            self.value_net = self._maybe_load_value_from_generic_checkpoint(checkpoint)

        if self.value_net is None and torch is not None and ValueNetwork is not None and self.rollout_weight < 1.0:
            self.value_net = ValueNetwork()
            self.value_net.to(self.device)
            self.value_net.eval()

        # 3. Khởi tạo Search Engine AlphaGoMCTS
        self.mcts = AlphaGoMCTS(
            policy_fn=self._policy_fn,
            value_fn=self._value_fn if self.value_net is not None else None,
            simulations=self.simulations,
            c_puct=self.c_puct,
            rollout_weight=self.rollout_weight if self.value_net is not None else 1.0,
            max_rollout_depth=self.max_rollout_depth,
            seed=self.seed,
        )

    def _load_policy_model(self, path: Union[str, Path]) -> Optional[Any]:
        """Nạp Policy Network từ file checkpoint theo hợp đồng HANDOFF.md."""
        if torch is None or CNNPolicyNetwork is None:
            return None
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Policy checkpoint không tồn tại: {ckpt_path}")

        try:
            payload = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except Exception as exc:
            raise ValueError(f"Không thể đọc checkpoint {ckpt_path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Định dạng checkpoint không hợp lệ ở {ckpt_path}")

        # Chỉ nạp nếu đúng là model CNNPolicyNetwork
        if payload.get("model_class") == "CNNPolicyNetwork":
            model_config = payload.get("model_config", {})
            model = CNNPolicyNetwork.from_config(model_config)
            model.load_state_dict(payload["state_dict"], strict=True)
            model.to(self.device)
            model.eval()
            return model

        return None

    def _load_value_model(self, path: Union[str, Path]) -> Optional[Any]:
        """Nạp Value Network từ file checkpoint theo hợp đồng HANDOFF.md."""
        if torch is None or ValueNetwork is None:
            return None
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Value checkpoint không tồn tại: {ckpt_path}")

        try:
            payload = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except Exception as exc:
            raise ValueError(f"Không thể đọc checkpoint {ckpt_path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Định dạng checkpoint không hợp lệ ở {ckpt_path}")

        if payload.get("model_class") == "ValueNetwork":
            model_config = payload.get("model_config", {})
            model = ValueNetwork.from_config(model_config)
            model.load_state_dict(payload["state_dict"], strict=True)
            model.to(self.device)
            model.eval()
            return model

        return None

    def _maybe_load_value_from_generic_checkpoint(self, path: Union[str, Path]) -> Optional[Any]:
        """Cố gắng nhận diện nếu checkpoint truyền vào là ValueNetwork."""
        if torch is None or ValueNetwork is None:
            return None
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            return None
        try:
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and payload.get("model_class") == "ValueNetwork":
                return self._load_value_model(ckpt_path)
        except Exception:
            return None
        return None

    def _policy_fn(self, board: ChessBoard) -> Dict[chess.Move, float]:
        """
        Dùng Policy Network tính phân phối xác suất tiên nghiệm P(s, a)
        cho các nước đi hợp lệ trên bàn cờ.
        """
        legal_moves = board.get_legal_actions()
        if not legal_moves:
            return {}

        # Nếu không có mạng nơ-ron hoặc PyTorch, chia đều xác suất
        if self.policy_net is None or torch is None:
            n = len(legal_moves)
            return {m: 1.0 / n for m in legal_moves}

        # 1. Mã hóa bàn cờ: HWC (8, 8, 14) -> NCHW (1, 14, 8, 8)
        hwc_state = ChessEncoder.encode_board(board.board)
        chw_state = np.transpose(hwc_state, (2, 0, 1)).copy()
        tensor = torch.from_numpy(chw_state).unsqueeze(0).to(self.device)

        # 2. Forward pass không tính gradient
        with torch.no_grad():
            logits = self.policy_net(tensor)[0]  # shape (4672,)

        # 3. Lấy logit cho từng nước đi hợp lệ
        move_logits: Dict[chess.Move, float] = {}
        turn = board.turn
        for move in legal_moves:
            action_idx = ChessEncoder.encode_move(move, turn)
            move_logits[move] = float(logits[action_idx].item())

        # 4. Softmax ổn định số học (trừ max logit)
        max_l = max(move_logits.values())
        exp_vals = {m: math.exp(l - max_l) for m, l in move_logits.items()}
        sum_exp = sum(exp_vals.values())

        if sum_exp <= 0 or math.isnan(sum_exp):
            n = len(legal_moves)
            return {m: 1.0 / n for m in legal_moves}

        return {m: v / sum_exp for m, v in exp_vals.items()}

    def _value_fn(self, board: ChessBoard) -> float:
        """
        Dùng Value Network lượng giá thế cờ: trả về điểm v trong [-1.0, 1.0]
        theo góc nhìn của bên đang tới lượt.
        """
        if self.value_net is None or torch is None:
            return 0.0

        hwc_state = ChessEncoder.encode_board(board.board)
        chw_state = np.transpose(hwc_state, (2, 0, 1)).copy()
        tensor = torch.from_numpy(chw_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            val = self.value_net(tensor)[0, 0].item()

        return float(np.clip(val, -1.0, 1.0))

    def act(self, board: ChessBoard) -> chess.Move:
        """
        Quyết định và trả về một chess.Move hợp lệ.
        Ném ValueError nếu thế cờ đã kết thúc.
        Không làm thay đổi bàn cờ hay lịch sử của bên gọi.
        """
        if board.is_terminal():
            raise ValueError(f"Không thể đi nước cờ trên bàn cờ đã kết thúc: {board.fen()}")

        legal_moves = board.get_legal_actions()
        if not legal_moves:
            raise ValueError("Không có nước đi hợp lệ nào trên bàn cờ")

        # Đảm bảo bàn cờ được clone an toàn trước khi tìm kiếm
        search_board = board.clone()
        chosen_move = self.mcts.search(search_board)

        if not board.is_legal(chosen_move):
            raise RuntimeError(f"AlphaGoAgent sinh ra nước đi phạm luật: {chosen_move}")

        return chosen_move

    def reset(self) -> None:
        """Đặt lại trạng thái bộ tìm kiếm giữa các ván cờ."""
        self.mcts.reset()
