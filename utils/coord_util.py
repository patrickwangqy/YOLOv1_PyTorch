import torch
from typing import Tuple

import config


def global_to_cell_coord(global_coord: torch.Tensor, pos_mask: torch.Tensor, cell_size: Tuple[int, int], box_num: int) -> torch.Tensor:
    """全局坐标转cell坐标

    Arguments:
        global_coord {torch.Tensor} -- shape: [..., (x, y, w, h)]，全局坐标

    Returns:
        torch.Tensor -- shape: [..., (x', y', sqrt(w), sqrt(h))]，cell坐标

    """
    offset_y = torch.arange(config.cell_size[0], dtype=torch.float32).reshape(-1, 1, 1).repeat(pos_mask.shape[0], 1, config.cell_size[1], box_num).to(config.device)
    offset_x = offset_y.permute([0, 2, 1, 3])

    global_coord = global_coord[pos_mask]
    offset_y = offset_y[pos_mask]
    offset_x = offset_x[pos_mask]

    global_x = global_coord[..., 0]
    global_y = global_coord[..., 1]
    global_w = global_coord[..., 2]
    global_h = global_coord[..., 3]

    cell_w = torch.sqrt(global_w)
    cell_h = torch.sqrt(global_h)

    cell_x = global_x * cell_size[1] - offset_x
    cell_y = global_y * cell_size[0] - offset_y

    cell_coord = torch.stack([cell_x, cell_y, cell_w, cell_h], dim=-1)

    return cell_coord


def cell_to_global_coord(cell_coord: torch.Tensor, pos_mask: torch.Tensor, cell_size: Tuple[int, int], box_num: int) -> torch.Tensor:
    """cell坐标转全局坐标

    Arguments:
        cell_coord {torch.Tensor} -- shape: [..., (x', y', sqrt(w), sqrt(h))]，cell坐标

    Returns:
        torch.Tensor -- shape: [..., (x, y, w, h)]，全局坐标
    """
    offset_y = torch.arange(config.cell_size[0], dtype=torch.float32).reshape(-1, 1, 1).repeat(pos_mask.shape[0], 1, config.cell_size[1], box_num).to(config.device)
    offset_x = offset_y.permute([0, 2, 1, 3])

    cell_coord = cell_coord[pos_mask]
    offset_y = offset_y[pos_mask]
    offset_x = offset_x[pos_mask]

    cell_x = cell_coord[..., 0]
    cell_y = cell_coord[..., 1]
    cell_w = cell_coord[..., 2]
    cell_h = cell_coord[..., 3]

    global_w = cell_w ** 2
    global_h = cell_h ** 2

    global_x = (cell_x + offset_x) / cell_size[1]
    global_y = (cell_y + offset_y) / cell_size[0]

    global_coord = torch.stack([global_x, global_y, global_w, global_h], dim=-1)
    return global_coord


def calc_iou(bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> torch.Tensor:
    """计算两组bbox的iou，两组bbox需要有相同的shape

    Arguments:
        bboxes1 {torch.Tensor} -- shape: [..., (cx, cy, w, h)]
        bboxes2 {torch.Tensor} -- shape: [..., (cx, cy, w, h)]

    Returns:
        torch.Tensor -- shape: [..., (iou)]
    """
    # cx,cy,w,h -> x1,y1,x2,y2
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
