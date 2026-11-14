import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


class PolicyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def train_supervised(data_path="logs/expert_data.pkl", save_path="checkpoints/policy_sl.pt", epochs=20):
    os.makedirs("checkpoints", exist_ok=True)
    with open(data_path, "rb") as f:
        X, y = pickle.load(f)

    X_tensor = torch.tensor(X).unsqueeze(1)
    y_tensor = torch.tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx).argmax(dim=1)
                correct += (preds == by).sum().item()
                total += by.size(0)

        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {correct/total:.2%}")

    torch.save(model.state_dict(), save_path)
    print(f"Đã lưu trọng số Policy Network tại {save_path}")


if __name__ == "__main__":
    train_supervised()