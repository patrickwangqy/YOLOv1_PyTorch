import torch

import config


def global_to_cell_coord(global_coord: torch.Tensor) -> torch.Tensor:
    """全局坐标转cell坐标

    Arguments:
        global_coord {torch.Tensor} -- 全局坐标
            [..., (y, x, h, w)]

    Returns:
        torch.Tensor -- cell坐标
            [..., (y', x', sqrt(h), sqrt(w))]
    """
    offset_y = torch.arange(config.cell_size[0], dtype=torch.float32).reshape(-1, 1, 1).repeat(1, config.cell_size[1], 1)
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
        cell_coord {torch.Tensor} -- cell坐标
            [..., (y', x', sqrt(h), sqrt(w))]

    Returns:
        torch.Tensor -- 全局坐标
            [..., (y, x, h, w)]
    """
    offset_y = torch.arange(config.cell_size[0], dtype=torch.float32).reshape(-1, 1, 1).repeat(1, config.cell_size[1], 1)
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
