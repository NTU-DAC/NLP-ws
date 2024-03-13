"""
PyTorch Example
"""
from typing import Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class TempDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y


    def __len__(self) -> int:
        return len(self.x)


    def __getitem__(self, idx: int) -> Tuple[float, float]:
        return self.x[idx], self.y[idx]


class MyMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(MyMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Trainer:
    def __init__(
            self, model: nn.Module, criterion: nn.Module,
            optimizer: torch.optim.Optimizer, epochs: int
        ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device('mps') if torch.backends.mps.is_available() \
            else torch.device('cpu')
        self.epochs = epochs
        self.model.to(self.device)


    def train(self, dataloader: DataLoader) -> None:
        tr_loss_lst, tr_acc_lst = [], []
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            train_loss, train_acc = [], []

            for batch in dataloader:
                feat, label = batch
                logits = self.model.forward(feat.to(self.device))
                loss = self.criterion(logits, label.to(self.device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                acc = (logits.argmax(1) == label.to(self.device)).float().mean()
                train_loss.append(loss.item())
                train_acc.append(acc)

            tr_loss = sum(train_loss) / len(train_loss)
            tr_acc = sum(train_acc) / len(train_acc)
            tr_loss_lst.append(tr_loss)
            tr_acc_lst.append(tr_acc)
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f}')


    def test(self, dataloader : DataLoader) -> torch.Tensor:
        for batch in tqdm(dataloader):
            pass
            


if __name__ == "__main__":
    torch.manual_seed(42)
    random_feature = torch.randn(10000, 10)
    random_label = torch.randint(0, 2, (10000, ))

    random_dataset = TempDataset(random_feature, random_label)
    random_dataloader = DataLoader(random_dataset, batch_size=32, shuffle=True)
    rand_tr_loader, random_ts_loader = random_split(random_dataloader, [8000, 2000])

    mlp = MyMLP(random_feature.size(1), 2)

    config = {
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": torch.optim.Adam(mlp.parameters(), lr=1e-3),
        "epochs": 30
    }

    trainer = Trainer(mlp, **config)
    trainer.train(rand_tr_loader)
            