import numpy as np
import torchvision

import config
from utils import image_util


class Validation(object):
    def __init__(self, model):
        self.model = model

    def val(self, dataset, transform):
        self.model.eval()
        for item in dataset:
            image, label = transform([item])
            to_show = np.array(item["image"])
            image = image.to(config.device)
            predicts = self.model(image)
            boxes, scores, classes = transform.decode(predicts)
            keep = torchvision.ops.nms(boxes, scores, 0.5)
            image_util.draw_boxes(to_show, boxes[keep], scores[keep], classes[keep], dataset.classes)
            # image_util.draw_predict(to_show, predicts[0], config.grid_size, config.boxes_num_per_cell)
            # image_util.draw_label(to_show, label[0], config.grid_size, config.boxes_num_per_cell)
            key = image_util.imshow(to_show)
            if key == 27:  # ESC
                break
