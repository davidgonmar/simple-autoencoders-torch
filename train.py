from training import train, TrainingConfig
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from models import FFAutoEncoder
import torch

DATA_ROOT = './data'

if __name__ == '__main__':
    torch.manual_seed(42)
    parser = TrainingConfig.add_argparse_args(argparse.ArgumentParser())
    args = parser.parse_args()
    config = TrainingConfig.from_args(args)

    model = FFAutoEncoder(input_size=(1, 28, 28), latent_dims=4)
    train_loader = DataLoader(MNIST(root=DATA_ROOT, train=True, download=True, transform=transforms.ToTensor()), batch_size=config.batch_size, shuffle=True)
    train(model, train_loader, config)