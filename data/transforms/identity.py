from typing import List
import torch
import numpy as np


def retrive_voc_bounding_box(bndbox, image_shape):
    return [int(bndbox["ymin"]) / image_shape[0], int(bndbox["xmin"]) / image_shape[1], int(bndbox["ymax"]) / image_shape[0], int(bndbox["xmax"]) / image_shape[1]]


class IdentityTransform(object):
    def __init__(self, classes_list: List):
        self.classes_list = classes_list

    def image_transform(self, image):
        return torch.from_numpy(np.array(image))

    def target_transform(self, label):
        image_shape = label["annotation"]["size"]
        image_shape = int(image_shape["height"]), int(image_shape["width"]), int(image_shape["depth"])
        objects = label["annotation"]["object"]
        if isinstance(objects, dict):
            objects = [objects]
        bboxes = list(map(lambda x: retrive_voc_bounding_box(x["bndbox"], image_shape), objects))
        class_ids = list(map(lambda x: self.classes_list.index(x["name"]), objects))
        labels = torch.cat([torch.tensor(bboxes, dtype=torch.float32), torch.unsqueeze(torch.tensor(class_ids, dtype=torch.float32), dim=-1)], dim=-1)
        return labels
