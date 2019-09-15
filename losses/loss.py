import torch
from typing import Tuple

import config
from utils import coord_util


def calc_mask(iou_predict_label: torch.Tensor, label_response: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算掩码

    Arguments:
        iou_predict_label {torch.Tensor} -- 预测框与ground truth框的iou，shape: (batch, cell_size, cell_size, 2)
        label_response {torch.Tensor} -- ground truth响应，shape: (batch, cell_size, cell_size, 1)

    Returns:
        Tuple[torch.Tensor, torch.Tensor] -- 有目标的掩码，无目标的掩码，shape: (batch, cell_size, cell_size, 2)
    """
    object_mask = torch.zeros_like(iou_predict_label, dtype=torch.float)
    object_mask_index = iou_predict_label.max(dim=-1, keepdim=True)[1]
    object_mask.scatter_(-1, object_mask_index, 1)
    object_mask = object_mask * label_response
    no_object_mask = 1.0 - object_mask
    return object_mask, no_object_mask


def calc_classes_loss_func(predict_classes: torch.Tensor, label_classes: torch.Tensor, label_response: torch.Tensor) -> torch.Tensor:
    """计算类别损失

    Arguments:
        predict_classes {torch.Tensor} -- 预测类别，shape: (batch, cell_size, cell_size, 20)
        label_classes {torch.Tensor} -- ground truth类别，shape: (batch, cell_size, cell_size, 20)
        label_response {torch.Tensor} -- ground true响应，shape: (batch, cell_size, cell_size, 1)

    Returns:
        torch.Tensor -- 类别损失，shape: None
    """
    delta = label_response * (predict_classes - label_classes)
    class_loss = (delta ** 2).sum(dim=[1, 2, 3]).mean()
    return class_loss


def calc_confidence_loss(predict_confidence: torch.Tensor, iou_predict_label: torch.Tensor, object_mask: torch.Tensor, no_object_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算置信度损失

    Arguments:
        predict_confidence {torch.Tensor} -- 预测置信度，shape: [batch, cell_size, cell_size, 1, 2]
        iou_predict_label {torch.Tensor} -- 预测框与ground truth框的iou，shape: (batch, cell_size, cell_size, 2)
        object_mask {torch.Tensor} -- 有目标的掩码，shape: (batch, cell_size, cell_size, 2)
        no_object_mask {torch.Tensor} -- 无目标的掩码，shape: (batch, cell_size, cell_size, 2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor] -- 有目标的置信度损失，无目标的置信度损失，shape: None
    """
    object_confidence_delta = object_mask * (predict_confidence - iou_predict_label)
    object_confidence_loss = (object_confidence_delta ** 2).sum(dim=(1, 2, 3)).mean()

    no_object_confidence_delta = no_object_mask * predict_confidence
    no_object_confidence_loss = (no_object_confidence_delta ** 2).sum(dim=(1, 2, 3)).mean()

    return object_confidence_loss, no_object_confidence_loss


def calc_coord_loss(cell_predict_bboxes: torch.Tensor, cell_label_bboxes: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
    """计算坐标损失

    Arguments:
        cell_predict_bboxes {torch.Tensor} -- 预测bbox的坐标，shape: [batch, cell_size, cell_size, (cy, cx, sqrt(h), sqrt(h)), 2]
        cell_label_bboxes {torch.Tensor} -- ground truth bbox的坐标，shape: [batch, cell_size, cell_size, , (cy, cx, sqrt(h), sqrt(h)), 2]
        object_mask {torch.Tensor} -- 有目标的掩码，shape: None

    Returns:
        torch.Tensor -- [description]
    """
    coord_mask = object_mask.unsqueeze(dim=-1)
    coord_delta = coord_mask * (cell_predict_bboxes - cell_label_bboxes)
    coord_loss = (coord_delta ** 2).sum(dim=(1, 2, 3, 4)).mean()
    return coord_loss


def yolo_loss(labels: torch.Tensor, predicts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算损失

    Arguments:
        labels {torch.Tensor} -- grount truth，shape: [batch, cell_size, cell_size, 25]
        predicts {torch.Tensor} -- 预测，shape: [batch, cell_size, cell_size, 30]

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] -- 类别损失，有目标的置信度损失，无目标的置信度损失，坐标损失，shape: None
    """
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
    predict_global_bboxes = coord_util.cell_to_global_coord(predict_bboxes)
    iou_predict_label = coord_util.calc_iou(predict_global_bboxes, label_bboxes)
    object_mask, no_object_mask = calc_mask(iou_predict_label, label_response)
    object_confidence_loss, no_object_confidence_loss = calc_confidence_loss(predict_confidences, iou_predict_label, object_mask, no_object_mask)
    label_cell_bboxes = coord_util.global_to_cell_coord(label_bboxes)
    coord_loss = calc_coord_loss(predict_bboxes, label_cell_bboxes, object_mask)
    return class_loss, object_confidence_loss, no_object_confidence_loss, coord_loss
