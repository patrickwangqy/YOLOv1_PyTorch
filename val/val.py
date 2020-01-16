import numpy as np

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
            image_util.draw_predict(to_show, predicts[0], config.cell_size, config.boxes_num_per_cell)
            image_util.draw_label(to_show, label[0], config.cell_size, config.boxes_num_per_cell)
            image_util.imshow(to_show)
