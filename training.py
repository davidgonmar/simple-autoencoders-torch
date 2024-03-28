from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    momentum: float
    weight_decay: float
    seed: int
    device: str

    @classmethod
    def from_args(cls, args):
        return cls(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device
        )

    @classmethod
    def add_argparse_args(cls, parser):
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
        return parser

def train(model: nn.Module, train_loader: DataLoader, config: TrainingConfig):
    model.to(config.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(config.epochs):
        model.train()
        for x, _ in train_loader:
            x = x.to(config.device)
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{config.epochs}, Loss: {loss.item()}')
    