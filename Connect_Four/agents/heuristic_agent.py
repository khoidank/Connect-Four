import random
from agents.base_agent import BaseAgent
from env.game_env import ConnectFourEnv


class HeuristicAgent(BaseAgent):
    def __init__(self, seed: int | None = None):
        super().__init__(name="Level 2 - Heuristic")
        self.rng = random.Random(seed)
        self.center_priority = [3, 2, 4, 1, 5, 0, 6]

    def _is_winning_move(self, env: ConnectFourEnv, action: int, player: int) -> bool:
        sim_env = env.clone()
        sim_env.current_player = player
        _, _, done, _ = sim_env.step(action)
        return done and sim_env.winner == player

    def select_action(self, env: ConnectFourEnv) -> int:
        legal = env.legal_actions()
        if len(legal) == 1:
            return legal[0]

        me = env.current_player
        opp = -me

        for col in legal:
            if self._is_winning_move(env, col, me):
                self.metrics = {"rule_triggered": "immediate_win"}
                return col

        for col in legal:
            if self._is_winning_move(env, col, opp):
                self.metrics = {"rule_triggered": "block_opponent"}
                return col

        safe_moves = []
        for col in legal:
            sim_env = env.clone()
            sim_env.step(col)
            opp_can_win = any(
                self._is_winning_move(sim_env, opp_col, opp)
                for opp_col in sim_env.legal_actions()
            )
            if not opp_can_win:
                safe_moves.append(col)

        candidates = safe_moves if safe_moves else legal

        for col in candidates:
            sim_env = env.clone()
            sim_env.step(col)
            win_threats = sum(
                1 for next_col in sim_env.legal_actions()
                if self._is_winning_move(sim_env, next_col, me)
            )
            if win_threats >= 2:
                self.metrics = {"rule_triggered": "double_attack"}
                return col

        for col in self.center_priority:
            if col in candidates:
                self.metrics = {"rule_triggered": "center_control"}
                return col

        return self.rng.choice(candidates)