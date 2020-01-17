from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import coord_util


class YOLOLoss(nn.Module):
    def __init__(self, grid_size: Tuple[int, int], box_num: int, class_num: int):
        """constract function

        Arguments:
            grid_size {Tuple[int, int]} -- cell rows, cell columns
            class_num {int} -- number of classes

        Keyword Arguments:
            box_num {int} -- number of boxes (default: {1})
        """
        super().__init__()
        self.grid_size = grid_size
        self.class_num = class_num
        self.box_num = box_num

    def forward(self, p: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """calculate loss

        Arguments:
            p {torch.Tensor} -- the predicts by model
            gt {torch.Tensor} -- the ground truth

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] -- class loss, obj conf loss, no obj conf loss, coord loss
        """
        # batch, cell_rows, cell_cols, box_num, 5
        p_loc = p[..., :5 * self.box_num].reshape(-1, *self.grid_size, self.box_num, 5)
        # batch, cell_rows, cell_cols, class_num
        p_class = p[..., 5 * self.box_num:]
        # batch, cell_rows, cell_cols, box_num
        p_conf = p_loc[..., 0]
        # batch, cell_rows, cell_cols, box_num, 4
        p_box = p_loc[..., 1:5]

        # batch, cell_rows, cell_cols, box_num, 5
        gt_loc = gt[..., :5 * self.box_num].reshape(-1, *self.grid_size, self.box_num, 5)
        # batch, cell_rows, cell_cols, class_num
        gt_class = gt[..., 5 * self.box_num:]
        # batch, cell_rows, cell_cols, box_num
        gt_response = gt_loc[..., 0]
        # batch, cell_rows, cell_cols, box_num, 4
        gt_box = gt_loc[..., 1:5]

        # batch, cell_rows, cell_cols, box_num
        box_mask = torch.zeros_like(p_conf, dtype=torch.bool)
        _, box_conf_max_idx = p_conf.max(keepdim=True, dim=-1)
        box_mask.scatter_(-1, box_conf_max_idx, True)
        # batch, cell_rows, cell_cols
        pos_mask = gt_response[..., 0] > 0
        # batch, cell_rows, cell_cols, box_num
        pos_cell_mask = pos_mask.unsqueeze(-1) & box_mask
        neg_cell_mask = ~pos_cell_mask

        class_loss = self.calc_class_loss(p_class, gt_class, pos_mask)
        xy_loss, wh_loss = self.calc_coord_loss(p_box, gt_box, pos_cell_mask)
        pos_conf_loss, neg_conf_loss = self.calc_conf_loss(p_conf, p_box, gt_box, pos_cell_mask, neg_cell_mask)

        loss = class_loss + 2 * pos_conf_loss + 0.5 * neg_conf_loss + 5 * xy_loss + 5 * wh_loss
        return loss, class_loss, xy_loss, wh_loss, pos_conf_loss, neg_conf_loss

    def calc_class_loss(self, p_class: torch.Tensor, gt_class: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        """calculate class loss

        Arguments:
            p_class {torch.Tensor} -- predict class one-hot, (batch, cell_rows, cell_cols, class_num)
            gt_class {torch.Tensor} -- ground truth class one-hot, (batch, cell_rows, cell_cols, class_num)
            pos_mask {torch.Tensor} -- cell mask where ground truth have object, (batch, cell_rows, cell_cols)

        Returns:
            torch.Tensor -- loss, (None,)
        """
        p_class = p_class[pos_mask]
        gt_class = gt_class[pos_mask]
        class_loss = F.mse_loss(p_class, gt_class)
        return class_loss

    def calc_coord_loss(self, p_box: torch.Tensor, gt_box: torch.Tensor, pos_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """calculate coord loss

        Arguments:
            p_box {torch.Tensor} -- predict class one-hot, (batch, cell_rows, cell_cols, box_num, 4)
            gt_box {torch.Tensor} -- ground truth class one-hot, (batch, cell_rows, cell_cols, box_num, 4)
            pos_mask {torch.Tensor} -- cell mask where ground truth have object, (batch, cell_rows, cell_cols, box_num)

        Returns:
            Tuple[torch.Tensor, torch.Tensor] -- xy loss, wh loss,  (None,) (None,)
        """
        p_box = p_box[pos_mask]
        p_xy = p_box[..., :2]
        p_wh = p_box[..., 2:]
        gt_box = coord_util.global_to_cell_coord(gt_box, pos_mask, self.grid_size, self.box_num)
        # gt_box = gt_box[pos_mask]
        gt_xy = gt_box[..., :2]
        gt_wh = gt_box[..., 2:]
        xy_loss = F.mse_loss(p_xy, gt_xy)
        wh_loss = F.mse_loss(p_wh, gt_wh)
        return xy_loss, wh_loss

    def calc_conf_loss(self,
                       p_conf: torch.Tensor,
                       p_box: torch.Tensor,
                       gt_box: torch.Tensor,
                       pos_mask: torch.Tensor,
                       neg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """calculate conf loss

        Arguments:
            p_conf {torch.Tensor} -- (batch, cell_rows, cell_cols, box_num)
            p_box {torch.Tensor} -- (batch, cell_rows, cell_cols, box_num, 4)
            gt_box {torch.Tensor} -- (batch, cell_rows, cell_cols, box_num, 4)
            pos_mask {torch.Tensor} -- (batch, cell_rows, cell_cols, box_num)
            neg_mask {torch.Tensor} -- (batch, cell_rows, cell_cols, box_num)

        Returns:
            Tuple[torch.Tensor, torch.Tensor] -- pos conf loss, neg conf loss
        """
        pos_p_conf = p_conf[pos_mask]
        with torch.no_grad():
            pos_p_box = coord_util.cell_to_global_coord(p_box, pos_mask, self.grid_size, self.box_num)
        neg_p_conf = p_conf[neg_mask]
        pos_gt_box = gt_box[pos_mask]
        with torch.no_grad():
            iou = coord_util.calc_iou(pos_p_box, pos_gt_box)
        pos_conf_loss = F.mse_loss(pos_p_conf, iou)
        neg_conf_loss = F.mse_loss(neg_p_conf, torch.zeros_like(neg_p_conf))
        return pos_conf_loss, neg_conf_loss
