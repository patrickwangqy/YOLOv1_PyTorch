from typing import Tuple, List, Dict, Union
import torch
from torchvision import transforms
from PIL.Image import Image

from utils import coord_util


mean = [0.4472466, 0.42313144, 0.39118877]
std = [0.2670601, 0.26384422, 0.27702332]


class YOLOTransform(object):
    def __init__(self,
                 image_size: Tuple[int, int],
                 grid_size: Tuple[int, int],
                 class_num: int):
        """YOLO Transform

        Arguments:
            image_size {Tuple[int, int]} -- image width, image height
            grid_size {Tuple[int, int]} -- grid rows, grid cols
            class_num {int} -- the class number
        """
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.grid_size = grid_size
        self.class_num = class_num

    def __call__(self, data: Union[Dict, List[Dict]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """do the transform

        Arguments:
            item {Dict} -- {"image": PIL image, "boxes": list of boxes, "labels": list of labels}

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]] -- the image and the ground truth
        """
        if isinstance(data, dict):
            data = [data]
        images = []
        targets = []
        for item in data:
            image: Image = item["image"]
            boxes = item["boxes"]
            labels = item["labels"]
            tensor_image = self.image_transform(image)
            target = self.target_transform((image.width, image.height), boxes, labels)
            images.append(tensor_image.unsqueeze(dim=0))
            targets.append(target.unsqueeze(dim=0))
        return torch.cat(images), torch.cat(targets)

    def target_transform(self, image_size: Tuple[int, int], boxes: List[List[int]], labels: List[int]):
        norm_xyxy = torch.tensor(boxes, dtype=torch.float)
        norm_xyxy[:, 0] /= image_size[0]
        norm_xyxy[:, 1] /= image_size[1]
        norm_xyxy[:, 2] /= image_size[0]
        norm_xyxy[:, 3] /= image_size[1]
        norm_xywh = torch.zeros_like(norm_xyxy)
        norm_xywh[:, 0] = (norm_xyxy[:, 0] + norm_xyxy[:, 2]) / 2.0
        norm_xywh[:, 1] = (norm_xyxy[:, 1] + norm_xyxy[:, 3]) / 2.0
        norm_xywh[:, 2] = norm_xyxy[:, 2] - norm_xyxy[:, 0]
        norm_xywh[:, 3] = norm_xyxy[:, 3] - norm_xyxy[:, 1]

        feature_size = 5 + self.class_num
        yolo_label = torch.zeros([*self.grid_size, feature_size], dtype=torch.float)
        for xywh, label in zip(norm_xywh, labels):
            x_index = int(xywh[0] * self.grid_size[0])
            y_index = int(xywh[1] * self.grid_size[1])
            if yolo_label[y_index, x_index, 0] == 1:
                continue
            yolo_label[y_index, x_index, 1:5] = xywh
            yolo_label[y_index, x_index, 0] = 1
            yolo_label[y_index, x_index, 5 + label] = 1
        return yolo_label

    def decode(self, predict: torch.Tensor, score: float, box_num: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        locs = predict[..., :5 * box_num].reshape(-1, self.grid_size[0], self.grid_size[1], box_num, 5)
        classes = predict[..., 5 * box_num:].unsqueeze(-2).repeat(1, 1, 1, box_num, 1)
        scores = locs[..., 0]
        boxes = locs[..., 1:5]
        mask = scores > score
        if mask.any():
            xywh = coord_util.cell_to_global_coord(boxes, mask, self.grid_size, box_num)
            xyxy = torch.zeros_like(xywh)
            xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
            xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
            xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
            xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2
            scores = scores[mask]
            classes = classes[mask]
            classes = torch.argmax(classes, dim=-1)
            return xyxy, scores, classes
        return None, None, None
