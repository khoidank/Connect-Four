import numpy as np
import torch
from agents.base_agent import BaseAgent
from env.game_env import ConnectFourEnv
from training.train_sl import PolicyCNN


class NNAgent(BaseAgent):
    def __init__(self, model_path: str = "checkpoints/policy_sl.pt", name: str = "Level 5 - Policy NN (SL)"):
        super().__init__(name=name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def get_action_probs(self, env: ConnectFourEnv) -> np.ndarray:
        state = env.get_canonical_state()
        tensor_x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor_x).squeeze(0)

        legal = env.legal_actions()
        mask = torch.full((7,), float("-inf"), device=self.device)
        mask[legal] = 0.0
        probs = torch.softmax(logits + mask, dim=0).cpu().numpy()
        return probs

    def select_action(self, env: ConnectFourEnv) -> int:
        probs = self.get_action_probs(env)
        action = int(np.argmax(probs))
        self.metrics = {"max_prob": round(float(probs[action]), 4)}
        return action