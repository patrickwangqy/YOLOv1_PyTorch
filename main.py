import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

import config
from data.dataloaders import VOCYoloDataLoader
from losses import yolo_loss
from train import Train
from val import Validation
from models import DetectModel


def train_exec(args):
    dataloader: DataLoader = VOCYoloDataLoader(args.data_root, "2007").build_dataloader("train", batch_size=16, shuffle=True)
    model: nn.Module = DetectModel().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train = Train(model, optimizer, yolo_loss, args.epochs, config.device)
    train.fit(dataloader, 10)


def val_exec(args):
    dataloader: DataLoader = VOCYoloDataLoader(args.data_root, "2007").build_dataloader("val", batch_size=1, shuffle=False, num_workers=0)
    model: nn.Module = DetectModel().to(config.device)
    model.load_state_dict(torch.load(args.checkpoint))
    val = Validation(model)
    val.val(dataloader)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="ssub command")

    # train
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train_exec)
    train_parser.add_argument("--data_root", type=str)
    train_parser.add_argument("--epochs", type=int, default=20)

    # val
    val_parser = subparsers.add_parser("val")
    val_parser.set_defaults(func=val_exec)
    val_parser.add_argument("--data_root", type=str)
    val_parser.add_argument("--checkpoint", type=str)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
