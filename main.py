import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

import config
from data.datasets import PascalVOCDataset as DataSet
from data.transforms import YOLOTransform
from losses import YOLOLoss
from train import Train
from val import Validation
from models import ResNet50 as Net


def train_exec(args):
    torch.manual_seed(config.SEED)
    dataset = DataSet(args.data_root)
    yolo_transform = YOLOTransform(config.image_size,
                                   config.grid_size,
                                   config.boxes_num_per_cell,
                                   len(dataset.classes))
    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            collate_fn=yolo_transform,
                            shuffle=True)
    model: nn.Module = Net(config.grid_size,
                           config.boxes_num_per_cell,
                           len(dataset.classes)).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    yolo_loss = YOLOLoss(config.grid_size,
                         config.boxes_num_per_cell,
                         len(dataset.classes))
    train = Train(model, optimizer, yolo_loss, args.epochs, config.device)
    train.fit(dataloader, 10)


def val_exec(args):
    dataset = DataSet(args.data_root)
    yolo_transform = YOLOTransform(config.image_size,
                                   config.grid_size,
                                   config.boxes_num_per_cell,
                                   len(dataset.classes))
    model: nn.Module = Net(config.grid_size,
                           config.boxes_num_per_cell,
                           len(dataset.classes)).to(config.device)
    model.load_state_dict(torch.load(args.checkpoint))
    val = Validation(model)
    val.val(dataset, yolo_transform)


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
