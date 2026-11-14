import numpy as np
from config import ROWS, COLS, CONNECT_N, EMPTY


class ConnectFourBoard:
    def __init__(self, grid: np.ndarray | None = None):
        if grid is not None:
            self.grid = grid.copy()
        else:
            self.grid = np.zeros((ROWS, COLS), dtype=np.int8)

    def get_legal_actions(self) -> list[int]:
        """Trả về danh sách các cột (0..6) còn trống ở hàng trên cùng."""
        return [c for c in range(COLS) if self.grid[0, c] == EMPTY]

    def is_valid_action(self, col: int) -> bool:
        """Kiểm tra cột thả quân có hợp lệ không."""
        return 0 <= col < COLS and self.grid[0, col] == EMPTY

    def drop_piece(self, col: int, player: int) -> int:
        """
        Thả quân cờ của `player` vào cột `col`.
        Trả về chỉ số hàng `row` mà quân cờ rơi xuống.
        """
        if not self.is_valid_action(col):
            raise ValueError(f"Nước đi không hợp lệ tại cột: {col}")

        # Tìm hàng trống thấp nhất (từ hàng 5 lên hàng 0)
        for r in range(ROWS - 1, -1, -1):
            if self.grid[r, col] == EMPTY:
                self.grid[r, col] = player
                return r
        raise RuntimeError("Lỗi: Cột hợp lệ nhưng không tìm thấy ô trống.")

    def check_win(self, last_row: int, last_col: int, player: int) -> bool:
        """
        Kiểm tra `player` có thắng ngay tại nước đi (last_row, last_col) không.
        Chỉ duyệt 4 trục qua ô vừa đánh để đạt tốc độ tối đa cho MCTS/Minimax.
        """
        directions = [
            (0, 1),   # Ngang
            (1, 0),   # Dọc
            (1, 1),   # Chéo chính
            (1, -1),  # Chéo phụ
        ]

        for dr, dc in directions:
            count = 1
            # Đếm về phía thuận
            r, c = last_row + dr, last_col + dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.grid[r, c] == player:
                count += 1
                r += dr
                c += dc

            # Đếm về phía ngược lại
            r, c = last_row - dr, last_col - dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.grid[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= CONNECT_N:
                return True
        return False

    def is_full(self) -> bool:
        """Kiểm tra bàn cờ đã đầy chưa (hòa nếu chưa ai thắng)."""
        return not np.any(self.grid[0, :] == EMPTY)

    def clone(self) -> "ConnectFourBoard":
        """Tạo bản sao nhanh mà không dùng deepcopy."""
        return ConnectFourBoard(grid=self.grid)