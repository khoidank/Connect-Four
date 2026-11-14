import numpy as np
from config import EMPTY, PLAYER_RED, PLAYER_YELLOW, ROWS, COLS
from env.board import ConnectFourBoard


class ConnectFourEnv:
    def __init__(self):
        self.board = ConnectFourBoard()
        self.current_player = PLAYER_RED
        self.done = False
        self.winner = None  # 1: Red thắng, -1: Yellow thắng, 0: Hòa, None: Chưa xong
        self.move_count = 0
        self.last_move = None  # Tuple (row, col)

    def reset(self) -> np.ndarray:
        """Đặt lại ván đấu về trạng thái ban đầu."""
        self.board = ConnectFourBoard()
        self.current_player = PLAYER_RED
        self.done = False
        self.winner = None
        self.move_count = 0
        self.last_move = None
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return self.board.grid.copy()

    def get_canonical_state(self) -> np.ndarray:
        return self.board.grid * self.current_player

    def legal_actions(self) -> list[int]:
        if self.done:
            return []
        return self.board.get_legal_actions()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("Ván đấu đã kết thúc. Hãy gọi reset() trước.")

        if action not in self.legal_actions():
            raise ValueError(f"Hành động {action} không hợp lệ. Các cột hợp lệ: {self.legal_actions()}")

        acting_player = self.current_player
        row = self.board.drop_piece(action, acting_player)
        self.last_move = (row, action)
        self.move_count += 1

        # Kiểm tra thắng/hòa
        if self.board.check_win(row, action, acting_player):
            self.done = True
            self.winner = acting_player
            reward = 1.0
        elif self.board.is_full():
            self.done = True
            self.winner = 0
            reward = 0.0
        else:
            reward = 0.0
            self.current_player *= -1  # Đổi lượt

        info = {
            "acting_player": acting_player,
            "last_move": self.last_move,
            "winner": self.winner,
            "move_count": self.move_count,
        }
        return self.get_state(), reward, self.done, info

    def is_terminal(self) -> bool:
        """Kiểm tra trò chơi đã kết thúc chưa."""
        return self.done

    def result(self, player: int = PLAYER_RED) -> float | None:
        if not self.done:
            return None
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def clone(self) -> "ConnectFourEnv":
        """Nhân bản nhanh môi trường phục vụ cây tìm kiếm Minimax / MCTS."""
        cloned = ConnectFourEnv.__new__(ConnectFourEnv)
        cloned.board = self.board.clone()
        cloned.current_player = self.current_player
        cloned.done = self.done
        cloned.winner = self.winner
        cloned.move_count = self.move_count
        cloned.last_move = self.last_move
        return cloned

    def render(self) -> None:
        """In trạng thái bàn cờ ra terminal."""
        symbols = {EMPTY: ".", PLAYER_RED: "X", PLAYER_YELLOW: "O"}
        print("\n  " + " ".join(str(c) for c in range(COLS)))
        print(" +" + "-" * (COLS * 2 - 1) + "+")
        for r in range(ROWS):
            row_str = " ".join(symbols[self.board.grid[r, c]] for c in range(COLS))
            print(f"{r}|" + row_str + "|")
        print(" +" + "-" * (COLS * 2 - 1) + "+")
        if self.done:
            if self.winner == PLAYER_RED:
                print("Kết quả: Quân Đỏ (X) THẮNG!")
            elif self.winner == PLAYER_YELLOW:
                print("Kết quả: Quân Vàng (O) THẮNG!")
            else:
                print("Kết quả: HÒA!")
        else:
            turn_name = "Đỏ (X)" if self.current_player == PLAYER_RED else "Vàng (O)"
            print(f"Lượt tiếp theo: {turn_name}")