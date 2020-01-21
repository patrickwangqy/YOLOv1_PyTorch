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


def draw_boxes(image, boxes, scores, class_ids, classes):
    x1s, y1s, x2s, y2s = boxes[..., 0].clamp(0, 1), boxes[..., 1].clamp(0, 1), boxes[..., 2].clamp(0, 1), boxes[..., 3].clamp(0, 1)
    colors = [(200, 0, 0),
              (200, 200, 0),
              (200, 0, 200),
              (0, 200, 0),
              (0, 200, 200),
              (0, 0, 200)]
    for i in range(boxes.shape[0]):
        x1 = int(x1s[i] * image.shape[1])
        y1 = int(y1s[i] * image.shape[0])
        x2 = int(x2s[i] * image.shape[1])
        y2 = int(y2s[i] * image.shape[0])
        color = colors[i % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{scores[i]:.2%}", (x1 + 5, y1 + 20), 1, 1, color)
        cv2.putText(image, f"{classes[class_ids[i]]}", (x1 + 5, y1 + 40), 1, 1, color)


def draw_label(image, result, grid_size, box_num):
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
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


def draw_predict(image, result, grid_size, box_num, threshold=0.7):
    result = result[..., :5 * box_num].reshape(grid_size[0], grid_size[1], box_num, 5)
    for i in range(config.grid_size[0]):
        for j in range(config.grid_size[1]):
            for k in range(config.boxes_num_per_cell):
                score = result[i, j, k, 0].item()
                if score < threshold:
                    continue
                cx = (result[i, j, k, 1].item() + j) / grid_size[1]
                cy = (result[i, j, k, 2].item() + i) / grid_size[0]
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
    try:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", image)
        return cv2.waitKey()
    finally:
        cv2.destroyWindow("image")
