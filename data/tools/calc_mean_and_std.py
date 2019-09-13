import numpy as np
from torch.utils.data.dataloader import DataLoader
import torchvision

import config
from data.transforms.yolo import YOLOTransform
from data.datasets.voc import VOCDataset


def get_mean_std(dataset, ratio=1.0):
    """Get mean and std by sample ratio
    """
    dataloader = DataLoader(dataset, batch_size=int(len(dataset) * ratio), shuffle=True, num_workers=0)
    for images, _labels in dataloader:
        mean = np.mean(images.numpy(), axis=(0, 2, 3))
        std = np.std(images.numpy(), axis=(0, 2, 3))
        break
    return mean, std


def main():
    voc_ds = VOCDataset("/mnt/data/datasets", "2007")
    yolo_trans = YOLOTransform(voc_ds.classes_list())
    train_ds = voc_ds.build_dataset("train", image_transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(config.image_size),
        torchvision.transforms.ToTensor()
    ]), target_transform=yolo_trans.target_transform)
    print(get_mean_std(train_ds))


if __name__ == "__main__":
    main()
