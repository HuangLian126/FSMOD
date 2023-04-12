# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from maskrcnn_benchmark.structures.image_list import to_image_list
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..backbone import build_backbone


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks. It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()


        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        label2vec_np = np.load('/home/hl/hl/maskrcnn-benchmark-4th/label2vec_glove_mod.npy')
        self.label2vec = nn.Parameter(torch.from_numpy(label2vec_np), requires_grad=False)

        self.fuse0 = nn.Sequential(
            torch.nn.Conv2d(256 * 2, 256, kernel_size=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True))

        self.fuse1 = nn.Sequential(
            torch.nn.Conv2d(256 * 2, 256, kernel_size=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True))

        self.fuse2 = nn.Sequential(
            torch.nn.Conv2d(256 * 2, 256, kernel_size=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True))

        self.fuse3 = nn.Sequential(
            torch.nn.Conv2d(256 * 2, 256, kernel_size=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True))

        self.fuse4 = nn.Sequential(
            torch.nn.Conv2d(256 * 2, 256, kernel_size=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True))


    def forward(self, images_c, images_t, gtMask=None, targets=None):

        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images_c = to_image_list(images_c)
        images_t = to_image_list(images_t)

        features_c, features_t = self.backbone([images_c.tensors, images_t.tensors])

        features = []
        for idx in range(len(features_c)):
            feature_c = features_c[idx]  # [b, c, H, W]
            feature_t = features_t[idx]  # [b, c, H, W]
            feature = eval('self.fuse' + str(idx))(torch.cat((feature_c, feature_t), dim=1))
            features.append(feature)
        features = tuple(features)


        semProto = self.label2vec
        # semProto = None

        proposals, proposal_losses = self.rpn(images_t, features, targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, semProto)
        else:
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            # losses.update(seg_losses)
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
