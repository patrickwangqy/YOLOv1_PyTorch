import torch

import config


def global_to_cell_coord(global_coord: torch.Tensor) -> torch.Tensor:
    """全局坐标转cell坐标

    Arguments:
        global_coord {torch.Tensor} -- shape: [..., (y, x, h, w)]，全局坐标

    Returns:
        torch.Tensor -- shape: [..., (y', x', sqrt(h), sqrt(w))]，cell坐标

    """
    offset_y = torch.arange(config.cell_size[0], dtype=torch.float32).reshape(-1, 1, 1).repeat(1, config.cell_size[1], 1).to(config.device)
    offset_x = offset_y.permute([1, 0, 2])

    global_y = global_coord[..., 0]
    global_x = global_coord[..., 1]
    global_h = global_coord[..., 2]
    global_w = global_coord[..., 3]

    cell_h = torch.sqrt(global_h)
    cell_w = torch.sqrt(global_w)

    cell_y = global_y * config.cell_size[0] - offset_y
    cell_x = global_x * config.cell_size[1] - offset_x

    cell_coord = torch.stack([cell_y, cell_x, cell_h, cell_w], dim=-1)

    return cell_coord


def cell_to_global_coord(cell_coord: torch.Tensor) -> torch.Tensor:
    """cell坐标转全局坐标

    Arguments:
        cell_coord {torch.Tensor} -- shape: [..., (y', x', sqrt(h), sqrt(w))]，cell坐标

    Returns:
        torch.Tensor -- shape: [..., (y, x, h, w)]，全局坐标
    """
    offset_y = torch.arange(config.cell_size[0], dtype=torch.float32).reshape(-1, 1, 1).repeat(1, config.cell_size[1], 1).to(config.device)
    offset_x = offset_y.permute([1, 0, 2])

    cell_y = cell_coord[..., 0]
    cell_x = cell_coord[..., 1]
    cell_h = cell_coord[..., 2]
    cell_w = cell_coord[..., 3]

    global_h = cell_h ** 2
    global_w = cell_w ** 2

    global_y = (cell_y + offset_y) / config.cell_size[0]
    global_x = (cell_x + offset_x) / config.cell_size[1]

    global_coord = torch.stack([global_y, global_x, global_h, global_w], dim=-1)
    return global_coord


def calc_iou(bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> torch.Tensor:
    """计算两组bbox的iou，两组bbox需要有相同的shape

    Arguments:
        bboxes1 {torch.Tensor} -- shape: [..., (cy, cx, h, w)]
        bboxes2 {torch.Tensor} -- shape: [..., (cy, cx, h, w)]

    Returns:
        torch.Tensor -- shape: [..., (iou)]
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
