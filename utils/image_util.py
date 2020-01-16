import cv2
import torch
import torchvision
import numpy as np

import config


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    for i, (mean, std) in enumerate(zip(config.mean, config.std)):
        tensor[i, :, :] = tensor[i, :, :] * std + mean
    image = torchvision.transforms.functional.to_pil_image(tensor)
    image = np.array(image).copy()
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def draw_bbox(image, bboxes):
    for i in range(bboxes.shape[0]):
        y1, x1, y2, x2 = bboxes[i, 0], bboxes[i, 1], bboxes[i, 2], bboxes[i, 3]
        x1 = int(x1 * image.shape[1])
        y1 = int(y1 * image.shape[0])
        x2 = int(x2 * image.shape[1])
        y2 = int(y2 * image.shape[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def draw_label(image, result, cell_size, box_num):
    for i in range(cell_size[0]):
        for j in range(cell_size[1]):
            for k in range(box_num):
                score = result[i, j, 0]
                if score > 0:
                    cx = result[i, j, 1]
                    cy = result[i, j, 2]
                    w = result[i, j, 3]
                    h = result[i, j, 4]
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    y1_ = int(y1 * image.shape[0])
                    x1_ = int(x1 * image.shape[1])
                    y2_ = int(y2 * image.shape[0])
                    x2_ = int(x2 * image.shape[1])
                    cv2.rectangle(image, (x1_, y1_), (x2_, y2_), (255, 0, 0), 2)


def draw_predict(image, result, cell_size, box_num, threshold=0.7):
    result = result[..., :5 * box_num].reshape(cell_size[0], cell_size[1], box_num, 5)
    for i in range(config.cell_size[0]):
        for j in range(config.cell_size[1]):
            for k in range(config.boxes_num_per_cell):
                score = result[i, j, k, 0].item()
                if score < threshold:
                    continue
                cx = (result[i, j, k, 1].item() + j) / cell_size[1]
                cy = (result[i, j, k, 2].item() + i) / cell_size[0]
                w = result[i, j, k, 3].item() ** 2
                h = result[i, j, k, 4].item() ** 2
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                x1_ = int(x1 * image.shape[1])
                y1_ = int(y1 * image.shape[0])
                x2_ = int(x2 * image.shape[1])
                y2_ = int(y2 * image.shape[0])
                cv2.rectangle(image, (x1_, y1_), (x2_, y2_), (0, 255, 0), 2)


def imshow(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyWindow("image")
