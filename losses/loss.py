import torch
from typing import Tuple

import config
from utils import coord_util


def calc_classes_loss_func(predict_classes, label_classes, label_response):
    """
    classes_predicts: predict (batch, cell_size, cell_size, 20)
    classes_labels: ground truth (batch, cell_size, cell_size, 20)
    labels_response: cell中是否有目标 (batch, cell_size, cell_size, 1)
    """
    delta = label_response * (predict_classes - label_classes)
    class_loss = torch.mean((delta ** 2).sum(dim=[1, 2, 3]))
    return class_loss


def calc_iou(bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> torch.Tensor:
    """
    输入两组bounding boxes，[..., 4]，代表cy, cx, h, w

    输出iou，[..., float]
    """
    # y,x,h,w -> y1,x1,y2,x2
    t_bboxes1 = torch.stack([
        bboxes1[..., 0] - bboxes1[..., 2] / 2.0,
        bboxes1[..., 1] - bboxes1[..., 3] / 2.0,
        bboxes1[..., 0] + bboxes1[..., 2] / 2.0,
        bboxes1[..., 1] + bboxes1[..., 3] / 2.0
    ], dim=-1)
    t_bboxes2 = torch.stack([
        bboxes2[..., 0] - bboxes2[..., 2] / 2.0,
        bboxes2[..., 1] - bboxes2[..., 3] / 2.0,
        bboxes2[..., 0] + bboxes2[..., 2] / 2.0,
        bboxes2[..., 1] + bboxes2[..., 3] / 2.0
    ], dim=-1)

    # top left
    tl = torch.max(t_bboxes1[..., :2], t_bboxes2[..., :2])
    # bottom right
    br = torch.min(t_bboxes1[..., 2:4], t_bboxes2[..., 2:4])

    area1 = bboxes1[..., 2] * bboxes1[..., 3]
    area2 = bboxes2[..., 2] * bboxes2[..., 3]

    intersection = torch.clamp(br - tl, min=0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1]

    union_area = torch.clamp(area1 + area2 - intersection_area, min=1e-8)

    iou = torch.clamp(intersection_area / union_area, min=0.0, max=1.0)
    return iou


def calc_mask(iou_predict_label, label_response):
    object_mask = iou_predict_label.max(dim=-1, keepdim=True)
    object_mask = (iou_predict_label >= object_mask).float()
    object_mask = object_mask * label_response
    no_object_mask = 1.0 - object_mask
    return object_mask, no_object_mask


def calc_confidence_loss(predict_confidence: torch.Tensor, iou_predict_label: torch.Tensor, object_mask: torch.Tensor, no_object_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    object_confidence_delta = object_mask * (predict_confidence - iou_predict_label)
    object_confidence_loss = (object_confidence_delta ** 2).sun(dim=(1, 2, 3)).mean()

    no_object_confidence_delta = no_object_mask * predict_confidence
    no_object_confidence_loss = (no_object_confidence_delta ** 2).sum(dim=(1, 2, 3)).mean()
    return object_confidence_loss, no_object_confidence_loss


def calc_coord_loss(predict_bboxes: torch.Tensor, label_bboxes: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
    coord_mask = object_mask.unsqueeze(dim=-1)
    coord_delta = coord_mask * (predict_bboxes - label_bboxes)
    coord_loss = (coord_delta ** 2).sum(dim=(1, 2, 3, 4)).mean()
    return coord_loss


def loss_func(labels, predicts):
    batch_size = predicts.shape[0]
    # shape: [batch, cell_size, cell_size, boxes_num_per_cell, 4]
    predict_bboxes = predicts[..., :4 * config.boxes_num_per_cell].reshape(batch_size, *config.cell_size, config.boxes_num_per_cell, 4)
    # shape: [batch, cell_size, cell_size, boxes_num_per_cell]
    predict_confidences = predicts[..., 4 * config.boxes_num_per_cell:5 * config.boxes_num_per_cell]
    # shape: [batch, cell_size, cell_size, classes_num]
    predict_classes = predicts[..., 5 * config.boxes_num_per_cell:]

    label_bboxes = labels[..., :4].reshape(batch_size, *config.cell_size, 1, 4).repeat(1, 1, 1, config.boxes_num_per_cell, 1)
    label_response = labels[..., 4].unsqueeze(dim=-1)
    label_classes = labels[..., 5:]

    class_loss = calc_classes_loss_func(predict_classes, label_classes, label_response)
    predict_global_bboxes = coord_util.cell_coord_to_global_coord(predict_bboxes)
    iou_predict_label = calc_iou(predict_global_bboxes, label_bboxes)
    object_mask, no_object_mask = calc_mask(iou_predict_label, label_response)
    object_confidence_loss, no_object_confidence_loss = calc_confidence_loss(predict_confidences, iou_predict_label, object_mask, no_object_mask)
    label_cell_bboxes = coord_util.global_coord_to_cell_coord(label_bboxes)
    coord_loss = calc_coord_loss(predict_bboxes, label_cell_bboxes, object_mask)
    return 1 * class_loss + 1 * object_confidence_loss + 0.5 * no_object_confidence_loss + 5 * coord_loss
