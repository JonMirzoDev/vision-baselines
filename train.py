"""Train an image-classification baseline on CIFAR-10.

Example:
    python train.py --model resnet18 --epochs 60
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_model

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(data_dir: str, batch_size: int, workers: int):
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=workers, pin_memory=True
    )
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == targets).sum().item()
    return correct / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lenet", choices=["lenet", "resnet18"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    train_loader, test_loader = build_loaders(args.data_dir, args.batch_size, args.workers)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    run_dir = Path("results") / (args.run_name or f"{args.model}-cifar10")
    run_dir.mkdir(parents=True, exist_ok=True)

    best_accuracy = 0.0
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_accuracy", "lr"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            scheduler.step()

            train_loss = running_loss / len(train_loader.dataset)
            accuracy = evaluate(model, test_loader, device)
            writer.writerow(
                [epoch, f"{train_loss:.4f}", f"{accuracy:.4f}", f"{scheduler.get_last_lr()[0]:.5f}"]
            )
            f.flush()
            print(f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  test acc {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), run_dir / "best.pt")

    print(f"best test accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
