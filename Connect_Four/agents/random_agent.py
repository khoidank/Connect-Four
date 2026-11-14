import random
from agents.base_agent import BaseAgent
from env.game_env import ConnectFourEnv


class RandomAgent(BaseAgent):
    def __init__(self, seed: int | None = None):
        super().__init__(name="Level 1 - Random")
        self.rng = random.Random(seed)

    def select_action(self, env: ConnectFourEnv) -> int:
        legal = env.legal_actions()
        if not legal:
            raise ValueError("Không còn nước đi hợp lệ.")
        self.metrics = {"legal_count": len(legal)}
        return self.rng.choice(legal)