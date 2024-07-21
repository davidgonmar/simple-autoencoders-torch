from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    momentum: float
    weight_decay: float
    seed: int
    device: str
    compile: bool
    save_path: str

    @classmethod
    def from_args(cls, args):
        return cls(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            compile=args.compile,
            save_path=args.save_path
        )

    @classmethod
    def add_argparse_args(cls, parser):
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--learning_rate', type=float, default=1) # will be adjusted by scheduler
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
        parser.add_argument('--compile', type=lambda x: x.lower() == 'true', default=True)
        parser.add_argument('--save_path', type=str, default='models/autoencoder.pth')
        return parser

def train(model: nn.Module, train_loader: DataLoader, config: TrainingConfig):
    model.to(config.device)
    if config.compile:
        model = torch.compile(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1., momentum=config.momentum, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    total_steps = len(train_loader) * config.epochs
    def lr_schedule(step):
        decay_factor = 0.85
        return decay_factor ** (step / (total_steps /30)) 
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc="Iterations", unit="batch")
        for x, _ in pbar:
            x = x.to(config.device)
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}')

    # save model (compiled has _orig_mod attribute)
    torch.save(model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict(), config.save_path)