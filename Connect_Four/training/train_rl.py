import os
import csv
import torch
from env.game_env import ConnectFourEnv
from training.train_sl import PolicyCNN
from agents.heuristic_agent import HeuristicAgent


def train_self_play_rl(
    init_weights: str = "checkpoints/policy_sl.pt",
    save_path: str = "checkpoints/policy_rl.pt",
    log_csv: str = "logs/rl_learning_curve.csv",
    episodes: int = 3000,
    gamma: float = 0.95,
):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyCNN().to(device)
    if os.path.exists(init_weights):
        model.load_state_dict(torch.load(init_weights, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    env = ConnectFourEnv()
    eval_opponent = HeuristicAgent(seed=42)

    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "loss", "win_rate_vs_heuristic"])

    for ep in range(1, episodes + 1):
        model.train()
        env.reset()
        trajectories = {1: [], -1: []}  # Lưu (log_prob, player)

        while not env.is_terminal():
            player = env.current_player
            state = torch.tensor(env.get_canonical_state(), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            logits = model(state).squeeze(0)

            legal = env.legal_actions()
            mask = torch.full((7,), float("-inf"), device=device)
            mask[legal] = 0.0
            probs = torch.softmax(logits + mask, dim=0)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            trajectories[player].append(dist.log_prob(action))

            env.step(int(action.item()))

        policy_loss = []
        for player in (1, -1):
            final_reward = env.result(player)
            G = final_reward
            returns = []
            for _ in reversed(trajectories[player]):
                returns.insert(0, G)
                G *= gamma
            for log_prob, r in zip(trajectories[player], returns):
                policy_loss.append(-log_prob * r)

        if policy_loss:
            optimizer.zero_grad()
            loss = torch.stack(policy_loss).sum()
            loss.backward()
            optimizer.step()

        if ep % 200 == 0:
            model.eval()
            wins = 0
            for i in range(40):
                env.reset()
                rl_side = 1 if i % 2 == 0 else -1
                while not env.is_terminal():
                    if env.current_player == rl_side:
                        st = torch.tensor(env.get_canonical_state(), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                        with torch.no_grad():
                            lg = model(st).squeeze(0)
                        mk = torch.full((7,), float("-inf"), device=device)
                        mk[env.legal_actions()] = 0.0
                        act = int(torch.argmax(lg + mk).item())
                    else:
                        act = eval_opponent.select_action(env)
                    env.step(act)
                if env.winner == rl_side:
                    wins += 1

            win_rate = wins / 40.0
            print(f"Episode {ep}/{episodes} | Win rate vs Heuristic: {win_rate:.1%}")
            with open(log_csv, "a", newline="") as f:
                csv.writer(f).writerow([ep, round(loss.item(), 4), win_rate])

    torch.save(model.state_dict(), save_path)
    print(f"Hoàn tất huấn luyện RL! Đã lưu tại {save_path}")


if __name__ == "__main__":
    train_self_play_rl()