# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (BalancedPositiveNegativeSampler)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN. Also supports FPN
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder, cls_agnostic_bbox_reg=False):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(matched_targets.bbox, proposals_per_image.bbox)

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def subsample(self, proposals, targets):
        labels, regression_targets = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)

        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(labels, regression_targets,
                                                                                       proposals):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field("regression_targets", regression_targets_per_image)

        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        self.target_ = targets
        return proposals

    def subsample_t(self, proposals_t, targets):
        labels_t, regression_targets_t = self.prepare_targets(proposals_t, targets)
        sampled_pos_inds_t, sampled_neg_inds_t = self.fg_bg_sampler(labels_t)

        proposals_t = list(proposals_t)

        for labels_per_image_t, regression_targets_per_image_t, proposals_per_image_t in zip(labels_t,
                                                                                             regression_targets_t,
                                                                                             proposals_t):
            proposals_per_image_t.add_field("labels", labels_per_image_t)
            proposals_per_image_t.add_field("regression_targets", regression_targets_per_image_t)

        for img_idx_t, (pos_inds_img_t, neg_inds_img_t) in enumerate(zip(sampled_pos_inds_t, sampled_neg_inds_t)):
            img_sampled_inds_t = torch.nonzero(pos_inds_img_t | neg_inds_img_t).squeeze(1)
            proposals_per_image_t = proposals_t[img_idx_t][img_sampled_inds_t]
            proposals_t[img_idx_t] = proposals_per_image_t

        self._proposals_t = proposals_t
        return proposals_t

    def __call__(self, class_logits, box_regression, xc_cpe_norm, semproto_cpe_norm, semProtoLabel):

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        #########################################计算分类损失##################################################
        proposals = self._proposals
        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat([proposal.get_field("regression_targets") for proposal in proposals], dim=0)
        classification_loss = F.cross_entropy(class_logits, labels)
        #####################################################################################################

        ####################################计算roi特征和语义原型之间的余弦相似度##################################
        fg_index = labels != 0
        xc_cpe_norm_fg = xc_cpe_norm[fg_index]  # [500, 128]
        labels_fg = labels[fg_index]  # [500]

        labels_trans = labels_fg.unsqueeze(1)  # [500, 1]
        semantic_label_trans = semProtoLabel.unsqueeze(0)  # [1,   6]
        label_mask = torch.eq(labels_trans, semantic_label_trans).float().cuda()  # [500, 6]

        similarity = torch.div(torch.matmul(xc_cpe_norm_fg, semproto_cpe_norm.t()), 1.0)  # [500, 6]
        #####################################################################################################

        ######################################计算roi特征和语义原型之间的度量损失##################################
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()
        exp_sim = torch.exp(similarity)

        log_prob = torch.log((exp_sim * label_mask).sum(1) / exp_sim.sum(1))
        SPM_loss = -log_prob
        SPM_loss = 0.5 * (SPM_loss.mean())
        #####################################################################################################


        #########################################计算回归损失##################################################
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1, )

        box_loss = box_loss / labels.numel()
        ######################################################################################################

        return classification_loss, box_loss, SPM_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
                                                    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION)

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
