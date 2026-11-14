from agents.nn_agent import NNAgent


class RLAgent(NNAgent):
    def __init__(self, model_path: str = "checkpoints/policy_rl.pt"):
        super().__init__(model_path=model_path, name="Level 6 - Self-Play RL")