# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    def __init__(self, grid_size, boxes_per_cell, categories, use_cuda=True):
        super(YOLOLoss, self).__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.categories = categories
        self.tl_coords = self._generate_grid_tl_coords()
        if use_cuda:
            self.tl_coords = self.tl_coords.cuda()

    def _generate_grid_tl_coords(self):
        grid_size_h, grid_size_w = self.grid_size
        y, x = np.ogrid[0:grid_size_h, 0:grid_size_w]
        y, x = y / grid_size_h, x / grid_size_w
        x = np.repeat(x, repeats=grid_size_h, axis=0)
        y = np.repeat(y, repeats=grid_size_w, axis=1)
        tl_coords = np.dstack([x, y]).astype(np.float32)
        tl_coords = torch.from_numpy(tl_coords).unsqueeze(dim=-2).repeat(1, 1, 1, 1)
        return tl_coords

    @staticmethod
    def compute_pos_object_confidence(bbox_a, bbox_b):
        assert bbox_a.size() == bbox_b.size() and bbox_a.size(1) == 4
        tl = torch.max(bbox_a[:, :2], bbox_b[:, :2])
        br = torch.min(bbox_a[:, 2:], bbox_b[:, 2:])
        area_i = torch.prod(br - tl, dim=1) * ((tl < br).all(dim=1).float())
        area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], dim=1)
        area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1)
        return area_i / (area_a + area_b - area_i)

    def _coord_loss(self, predict, ground_truth, pos_mask):
        predict_x_y = predict[..., :2] - self.tl_coords
        ground_truth_x_y = ground_truth[..., :2] - self.tl_coords
        predict_w_h = torch.sqrt(predict[..., 2:])
        ground_truth_w_h = torch.sqrt(ground_truth[..., 2:])

        predict_x_y = predict_x_y[pos_mask]
        ground_truth_x_y = ground_truth_x_y[pos_mask]
        predict_w_h = predict_w_h[pos_mask]
        ground_truth_w_h = ground_truth_w_h[pos_mask]

        xy_loss = 5 * F.mse_loss(predict_x_y, ground_truth_x_y)
        wh_loss = 5 * F.mse_loss(predict_w_h, ground_truth_w_h)
        return xy_loss, wh_loss

    def _confidence_loss(self, predict, ground_truth, pos_mask, neg_mask):
        pos_predict = predict[pos_mask]  # Mx5
        neg_predict = predict[neg_mask]  # Nx5
        pos_ground_truth = ground_truth[pos_mask]  # Mx5
        neg_ground_truth = ground_truth[neg_mask]  # Nx5

        # convert to xyxy mode
        p_c_x, p_c_y, p_w, p_h = pos_predict[:, 1:5].split(1, dim=1)
        predict_loc = torch.cat([
            p_c_x - p_w / 2,
            p_c_y - p_h / 2,
            p_c_x + p_w / 2,
            p_c_y + p_h / 2], dim=-1)
        g_c_x, g_c_y, g_w, g_h = pos_ground_truth[:, 1:5].split(1, dim=1)
        ground_truth_loc = torch.cat([
            g_c_x - g_w / 2,
            g_c_y - g_h / 2,
            g_c_x + g_w / 2,
            g_c_y + g_h / 2], dim=-1)
        # compute predict and ground truth iou
        g_pos_confidence = self.compute_pos_object_confidence(
            predict_loc, ground_truth_loc)
        p_pos_confidence = pos_predict[:, 0]
        pos_confidence_loss = F.mse_loss(p_pos_confidence, g_pos_confidence)

        g_neg_confidence = neg_ground_truth[:, 0]
        p_neg_confidence = neg_predict[:, 0]
        neg_confidence_loss = 0.5 * F.mse_loss(p_neg_confidence, g_neg_confidence)
        return pos_confidence_loss, neg_confidence_loss

    def _class_loss(self, predict, ground_truth, pos_mask):
        p_class = predict[pos_mask]
        g_class = ground_truth[pos_mask]
        loss = F.mse_loss(p_class, g_class)
        return loss

    def forward(self, predict, ground_truth):
        pos_mask = ground_truth[..., 0] > 0   # batch_size x 7 x 7
        neg_mask = ground_truth[..., 0] == 0  # batch_size x 7 x 7

        ground_truth_boxes = ground_truth[..., :5]
        ground_truth_boxes = ground_truth_boxes.unsqueeze(dim=-2).repeat(1, 1, 1, self.boxes_per_cell, 1)

        predict_boxes = predict[..., :5*self.boxes_per_cell]
        predict_boxes = predict_boxes.view(*predict_boxes.shape[:-1], self.boxes_per_cell, 5)

        box_conf = predict_boxes[..., 0]
        box_mask = torch.zeros_like(box_conf, dtype=torch.bool)
        _, box_conf_max_idx = box_conf.max(keepdim=True, dim=-1)
        box_mask.scatter_(-1, box_conf_max_idx, True)

        boxes_pos_mask = pos_mask.unsqueeze(-1) & box_mask
        boxes_neg_mask = neg_mask.unsqueeze(-1) & box_mask

        xy_loss, wh_loss = self._coord_loss(
            predict_boxes[..., 1:5], ground_truth_boxes[..., 1:5], boxes_pos_mask)
        pos_confidence_loss, neg_confidence_loss = self._confidence_loss(
            predict_boxes, ground_truth_boxes, boxes_pos_mask, boxes_neg_mask)

        class_loss = self._class_loss(predict[..., 5*self.boxes_per_cell:], ground_truth[..., 5:], pos_mask)
        loss = xy_loss + wh_loss + pos_confidence_loss + neg_confidence_loss + class_loss
        return loss, class_loss, xy_loss, wh_loss, pos_confidence_loss, neg_confidence_loss
