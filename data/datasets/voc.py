from torchvision.datasets import VOCDetection


class VOCDataset(object):
    def __init__(self, root_dir, year):
        self.root_dir = root_dir
        self.year = year

    def classes_list(self):
        return [
            "person",
            "bird", "cat", "cow", "dog", "horse", "sheep",
            "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
            "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"
        ]

    def build_dataset(self, image_set, image_transform=None, target_transform=None):
        voc_dataset = VOCDetection(self.root_dir, year=self.year, image_set=image_set, transform=image_transform, target_transform=target_transform)
        return voc_dataset
