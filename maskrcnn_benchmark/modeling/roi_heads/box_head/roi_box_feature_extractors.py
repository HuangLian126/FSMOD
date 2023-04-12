# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)

        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)

        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        xc = F.relu(self.fc6(x))
        xc = F.relu(self.fc7(xc))

        xr = F.relu(self.fc6r(x))
        xr = F.relu(self.fc7r(xr))

        return xc, xr

class FPN2MLPFeatureExtractor_each(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor_each, self).__init__()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio,)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler

        pooler_t = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio,)
        self.pooler_t = pooler_t

        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)

        self.fc6_t = make_fc(input_size, representation_size, use_gn)
        self.fc7_t = make_fc(representation_size, representation_size, use_gn)

        self.out_channels = representation_size

        # self.spatialAlign = spatialAlign()

    def forward(self, x, x_t, proposals):
        x   = self.pooler(x,     proposals)
        x_t = self.pooler_t(x_t, proposals)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        x_t = x_t.view(x_t.size(0), -1)
        x_t = F.relu(self.fc6_t(x_t))
        x_t = F.relu(self.fc7_t(x_t))

        return x, x_t

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor_single")
class FPN2MLPFeatureExtractor_single(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor_single, self).__init__()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio,)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler

        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)

        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor_single_spmLoss")
class FPN2MLPFeatureExtractor_single_spmLoss(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor_single_spmLoss, self).__init__()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio,)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler

        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)

        self.out_channels = representation_size

        self.head_xc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )

        self.head_PM = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward(self, x, proposals, semanticPrototype):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        x_cpe = self.head_xc(x)  # [512, 128]
        x_cpe_norm = F.normalize(x_cpe, dim=1)  # [512, 128]

        sp_cpe = self.head_PM(semanticPrototype)  # [7, 128]
        sp_cpe_norm = F.normalize(sp_cpe, dim=1)  # [7, 128]

        return x, x_cpe_norm, sp_cpe_norm

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor_dual")
class FPN2MLPFeatureExtractor_dual(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor_dual, self).__init__()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio,)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler

        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)

        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)

        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        xc = F.relu(self.fc6c(x))
        xc = F.relu(self.fc7c(xc))

        xr = F.relu(self.fc6r(x))
        xr = F.relu(self.fc7r(xr))

        return xc, xr

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor_dual_SPMLoss")
class FPN2MLPFeatureExtractor_dual_SPMLoss(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor_dual_SPMLoss, self).__init__()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio,)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler

        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)

        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)

        self.out_channels = representation_size

        self.head_xc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )

        self.head_PM = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )


    def forward(self, x, proposals, semanticPrototype):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        xc = F.relu(self.fc6c(x))
        xc = F.relu(self.fc7c(xc))                # [512, 1024]

        xc_cpe = self.head_xc(xc)                 # [512, 128]
        xc_cpe_norm = F.normalize(xc_cpe, dim=1)  # [512, 128]

        pm_cpe = self.head_PM(semanticPrototype)  # [7, 128]
        pm_cpe_norm = F.normalize(pm_cpe, dim=1)  # [7, 128]

        xr = F.relu(self.fc6r(x))
        xr = F.relu(self.fc7r(xr))

        return xc, xr, xc_cpe_norm, pm_cpe_norm

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x

def make_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR]
    return func(cfg, in_channels)