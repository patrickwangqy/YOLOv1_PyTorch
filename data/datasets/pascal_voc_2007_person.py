# -*- coding: utf-8 -*-

import os
from xml.etree.ElementTree import parse
from PIL import Image
import torch.utils.data


class PascalVOC2007Person(torch.utils.data.Dataset):
    def __init__(self, data_dir, use_difficult=True):
        self.image_folder = os.path.join(data_dir, "JPEGImages")
        self.annotation_folder = os.path.join(data_dir, "Annotations")
        self.titles = [
            name.replace(".xml", "")
            for name in os.listdir(self.annotation_folder) if ".xml" in name
        ]
        self.use_difficult = use_difficult
        self.classes = ["person"]

    @staticmethod
    def read_image(image_path):
        return Image.open(image_path).convert("RGB")

    def read_annotation_xml_file(self, annotation_file_path, use_difficult=False):
        boxes, labels = list(), list()
        root = parse(annotation_file_path).getroot()
        # iterative all objects, get boxes and labels
        for obj in root.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if difficult and use_difficult is False:
                continue
            name = obj.find("name").text.lower().strip()
            label = self.classes.index(name)
            bounding_box_node = obj.find("bndbox")
            # minus 1 for 0 based
            x_min = int(bounding_box_node.find("xmin").text) - 1
            y_min = int(bounding_box_node.find("ymin").text) - 1
            x_max = int(bounding_box_node.find("xmax").text) - 1
            y_max = int(bounding_box_node.find("ymax").text) - 1
            labels.append(label)
            boxes.append([x_min, y_min, x_max, y_max])
        return boxes, labels

    def __getitem__(self, index):
        title = self.titles[index]
        image_path = os.path.join(self.image_folder, F"{title}.jpg")
        annotation_file_path = os.path.join(self.annotation_folder,
                                            F"{title}.xml")
        image = self.read_image(image_path)
        boxes, labels = self.read_annotation_xml_file(annotation_file_path,
                                                      self.use_difficult)
        return dict(image=image, boxes=boxes, labels=labels)

    def __len__(self):
        return len(self.titles)
