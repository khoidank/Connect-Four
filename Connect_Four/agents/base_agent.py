from abc import ABC, abstractmethod
from env.game_env import ConnectFourEnv


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.metrics = {}  # Lưu thông số phụ trợ: nodes_expanded, rollouts, etc.

    @abstractmethod
    def select_action(self, env: ConnectFourEnv) -> int:
        pass

    def get_metrics(self) -> dict:
        return self.metrics.copy()