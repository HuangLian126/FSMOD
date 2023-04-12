# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F
from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

class ROIBoxHead(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None, semProto=None):

        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        ##############################################################################################
        xc, xc_cpe_norm, sp_cpe_norm = self.feature_extractor(features, proposals, semProto)
        # xc = self.feature_extractor(features, proposals)
        class_logits, box_regression = self.predictor(xc)
        ##############################################################################################

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals, proposals)
            return xc, result, {}

        semProtoLabel = torch.LongTensor([int(1), int(2), int(3), int(4), int(5), int(6)])
        semProtoLabel = semProtoLabel.cuda()

        loss_classifier, loss_box_reg, loss_spm = self.loss_evaluator([class_logits], [box_regression], xc_cpe_norm, sp_cpe_norm, semProtoLabel)
        # loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression])

        return (xc, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_spm=loss_spm))

def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class and make it a parameter in the config.
    """
    return ROIBoxHead(cfg, in_channels)