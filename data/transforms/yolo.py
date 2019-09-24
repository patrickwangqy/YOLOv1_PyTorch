from typing import List
import torch
import torchvision.transforms as transforms

import config


def retrive_voc_bounding_box(bndbox, image_shape):
    return [int(bndbox["ymin"]) / image_shape[0], int(bndbox["xmin"]) / image_shape[1], int(bndbox["ymax"]) / image_shape[0], int(bndbox["xmax"]) / image_shape[1]]


class YOLOTransform(object):
    def __init__(self, classes_list: List, image_size=config.image_size, cell_size=config.cell_size, box_num_per_cell=config.boxes_num_per_cell):
        self.image_size = image_size
        self.cell_size = cell_size
        self.box_num_per_cell = box_num_per_cell
        self.class_num = len(classes_list)
        self.classes_list = classes_list

        self.image_transform_obj = transforms.Compose([
            transforms.Resize([self.image_size[0], self.image_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ])

    def image_transform(self, image):
        return self.image_transform_obj(image)

    def target_transform(self, label):
        image_shape = label["annotation"]["size"]
        image_shape = int(image_shape["height"]), int(image_shape["width"]), int(image_shape["depth"])
        objects = label["annotation"]["object"]
        if isinstance(objects, dict):
            objects = [objects]
        bboxes = list(map(lambda x: retrive_voc_bounding_box(x["bndbox"], image_shape), objects))
        class_ids = list(map(lambda x: self.classes_list.index(x["name"]), objects))

        yolo_label = torch.zeros([*self.cell_size, 25], dtype=torch.float32)
        for bbox, class_id in zip(bboxes, class_ids):
            y1, x1, y2, x2 = bbox
            cy = (y1 + y2) / 2
            cx = (x1 + x2) / 2
            h = y2 - y1
            w = x2 - x1
            y_index = int(cy * self.cell_size[0])
            x_index = int(cx * self.cell_size[1])
            if yolo_label[y_index, x_index, 4] == 1:
                continue
            yolo_label[y_index, x_index, 4] = 1
            yolo_label[y_index, x_index, :4] = torch.tensor([cy, cx, h, w])
            yolo_label[y_index, x_index, 5 + class_id] = 1
        return yolo_label
