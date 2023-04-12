# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))

        images   = to_image_list(transposed_batch[0], self.size_divisible)
        images_t = to_image_list(transposed_batch[1], self.size_divisible)
        # gtMask = to_image_list(transposed_batch[2], self.size_divisible)

        gtMask   = transposed_batch[2]
        targets  = transposed_batch[3]
        img_ids  = transposed_batch[4]

        return images, images_t, gtMask, targets, img_ids

class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))