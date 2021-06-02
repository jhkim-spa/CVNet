import copy
import numpy as np
import torch
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply
from mmcv.cnn import bias_init_with_prob, normal_init

from mmdet3d.cv_utils import project, project_to_image, pad, resize


@HEADS.register_module()
class CenterHeadCV(nn.Module):
    def __init__(self,
                 num_classes=1,
                 feat_channels=64,
                 train_cfg=None,
                 test_cfg=None,
                 loss_cls=dict(type='FocalLoss',
                               use_sigmoid=True,
                               gamma=2.0,
                               alpha=0.25,
                               loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1loss',
                                beta=1.0 / 9.0,
                                loss_weight=2.0)):
        super(CenterHeadCV, self).__init__()

        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.feat_channels = feat_channels
        self.use_sigmoid_cls = loss_cls.use_sigmoid

        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.reg_target_size = 8

        self._init_layers()

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.cls_out_channels = self.num_classes
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.reg_target_size, 1)

    def init_weights(self):
        """Initialize the weights of head."""
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        """Forward function on a single-scale feature map.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            tuple[torch.Tensor]: Contain score of each class, bbox \
                regression and direction classification predictions.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)

        return cls_score, bbox_pred
        
    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple[list[torch.Tensor]]: Multi-level class score, bbox \
                and direction predictions.
        """
        return multi_apply(self.forward_single, feats)

    def loss_single(self, cls_score, bbox_pred, cls_targets, reg_targets):
        # classification loss
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        cls_targets = cls_targets.permute(0, 2, 3, 1).reshape(-1).to(torch.long)

        loss_cls = self.loss_cls(cls_score, cls_targets)

        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, self.reg_target_size)
        reg_targets = reg_targets.permute(0, 2, 3,
                                        1).reshape(-1, self.reg_target_size)

        pos_inds = (cls_targets == 1).reshape(-1)
        num_pos = pos_inds.sum()

        pos_bbox_pred = bbox_pred[pos_inds]
        pos_reg_targets = reg_targets[pos_inds]

        if num_pos > 0:
            loss_bbox = self.loss_bbox(pos_bbox_pred,
                                       pos_reg_targets,
                                       avg_factor=num_pos)

        else:
            loss_bbox = pos_bbox_pred.sum()

        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             input_metas,
             cv_size,
             pad_size,
             gt_bboxes_ignore=None):
        """Calculate losses.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]): Gt bboxes
                of each sample.
            gt_labels (list[torch.Tensor]): Gt labels of each sample.
            input_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and \
                direction losses of each level.

                - loss_cls (list[torch.Tensor]): Classification losses.
                - loss_bbox (list[torch.Tensor]): Box regression losses.
                - loss_dir (list[torch.Tensor]): Direction classification \
                    losses.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            device,
            self.num_classes,
            gt_bboxes,
            input_metas,
            cv_size,
            pad_size,
            featmap_sizes,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)

        if cls_reg_targets is None:
            return None
        (cls_targets_list, reg_targets_list) = cls_reg_targets

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            cls_targets_list,
            reg_targets_list)
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox)
    
    def get_targets(self,
            device,
            num_classes,
            gt_bboxes,
            input_metas,
            cv_size,
            pad_size,
            featmap_sizes,
            gt_bboxes_ignore_list,
            gt_labels_list,
            label_channels):
        norm = torch.tensor([70.4, 80, 4, 1.6, 3.9, 1.56], device=device)
        valid_idxs= [torch.where((res != -1) & (res < num_classes))[0]\
            for res in gt_labels_list]
        gt_bboxes = [box[idx].to(device) for box, idx in zip(gt_bboxes, valid_idxs)]
        gt_labels_list= [label[idx].to(device) for label, idx in zip(gt_labels_list, valid_idxs)]
        proj_mats = [torch.tensor(res['lidar2img'][:3]).to(device)\
            for res in input_metas]
        centers_3d = [res.gravity_center for res in gt_bboxes]
        centers_2d = [project_to_image(res.transpose(1, 0), proj_mat).to(torch.long)\
            for res, proj_mat in zip(centers_3d, proj_mats)]
        # centers_2d_ = [project(res, meta) for res, meta in zip(centers_3d, input_metas)]
        # centers_2d_ = [torch.nonzero(res[..., 0])[:, [1, 0]].permute(1, 0) for res in centers_2d_]
        ## shift x coords (padding)
        centers_2d = [res + torch.tensor([pad_size[0], 0], device=device).reshape(-1, 1)\
            for res in centers_2d]
        for i, centers in enumerate(centers_2d):
            if (centers < 0).sum() != 0:
                valid_idx = (0 <= centers[0]) &\
                            (centers[0] <= cv_size[1]) &\
                            (0 <= centers[1]) &\
                            (centers[1] <= cv_size[0])
                gt_labels_list[i] = gt_labels_list[i][valid_idx]
                gt_bboxes[i] = gt_bboxes[i][valid_idx]
                centers_2d[i] = centers_2d[i][:, valid_idx]
        gt_labels_list
        targets = [torch.cat((center.transpose(1, 0).to(torch.float32),
                              label.reshape(-1, 1).to(torch.float32),
                              box.tensor[:, :-1] / norm,
                              torch.cos(box.tensor[:, -1].reshape(-1, 1)),
                              torch.sin(box.tensor[:, -1].reshape(-1, 1))), dim=1)\
            for label, center, box in zip(gt_labels_list, centers_2d, gt_bboxes)]
        target_maps = []
        target_map_channel = label_channels + self.reg_target_size
        for target in targets:
            target_map = torch.zeros((cv_size[0], cv_size[1],
                target_map_channel), dtype=torch.float32, device=device)
            x_coords = target[:, 0].to(torch.long)
            y_coords = target[:, 1].to(torch.long)
            target = target[:, 2:]
            target_map[y_coords, x_coords, label_channels:] =\
                target[:, label_channels:]
            target_map[y_coords, x_coords, target[:, 0].to(torch.long)] = 1.
            target_maps.append(target_map)

        mlvl_target_maps = []
        for featmap_size in featmap_sizes:
            des_size = (featmap_size[1], featmap_size[0])
            target_maps_resized = [resize(res, des_size, nonzero_idx=1)\
                for res in target_maps]
            mlvl_target_maps.append(target_maps_resized)
        
        cls_targets = [[res[..., :label_channels].permute(2, 0, 1)\
            for res in target_maps] for target_maps in mlvl_target_maps]
        reg_targets = [[res[..., label_channels:label_channels +\
            self.reg_target_size].permute(2, 0, 1)\
            for res in target_maps] for target_maps in mlvl_target_maps]
        
        # stack batches
        cls_targets = [torch.stack(tuple(res), dim=0)\
            for res in cls_targets]
        reg_targets = [torch.stack(tuple(res), dim=0)\
            for res in reg_targets]

        return (cls_targets, reg_targets)
