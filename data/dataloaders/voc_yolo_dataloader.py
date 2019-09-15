from torch.utils.data.dataloader import DataLoader

from data.datasets.voc import VOCDataset
from data.transforms.yolo import YOLOTransform


class VOCYoloDataLoader(object):
    def __init__(self, root_dir, year):
        self.dataset = VOCDataset(root_dir, year)

    def build_dataloader(self, image_set, batch_size,
                         image_transform=True, target_transform=True,
                         shuffle=False, num_workers=8):
        transform = YOLOTransform(self.dataset.classes_list())
        image_transform = transform.image_transform if image_transform else None
        target_transform = transform.target_transform if target_transform else None
        dataset = self.dataset.build_dataset(image_set, image_transform, target_transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return data_loader
