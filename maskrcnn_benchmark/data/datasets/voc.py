import os
import torch
import torch.utils.data
from PIL import Image
import sys
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList

class PascalVOCDataset(torch.utils.data.Dataset):
    # CLASSES = ("__background__ ", "person")
    CLASSES = ("__background__ ", "bike", "car", "person", "carstop", "guardrail", "colorcone",)

    def __init__(self, data_dir, split, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages_rgb", "%s.png")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")
        self._imgpath_t = os.path.join("datasets/voc/VOC2012/JPEGImages_t", "%s.png")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

    def __getitem__(self, index):
        img_id = self.ids[index]
        # print(self._imgpath % img_id)

        img   = Image.open(self._imgpath % img_id).convert("RGB")
        img_t = Image.open(self._imgpath_t % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        # gtMask = torch.zeros(1, img.size[0], img.size[1])

        gtMask = Image.new('L', (img.size[0], img.size[1]), 0)
        gtMask = np.array(gtMask)

        for i in range(target.bbox.shape[0]):
            bbox_i = target.bbox[i]
            # print(bbox_i)
            gtMask[int(bbox_i[1]):int((bbox_i[3])), int((bbox_i[0])):int((bbox_i[2]))] = 255

        gtMask = Image.fromarray(np.uint8(gtMask))

        if self.transforms is not None:
            img, img_t, gtMask, target = self.transforms(img, img_t, gtMask, target)
            # gtMask1 = torch.zeros(1, img.shape[1], img.shape[2])

        # img_cat = torch.cat((img, img_t), dim=0)
        # img_cat = [img, img_t]

        return img, img_t, gtMask, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(map(lambda x: x - TO_REMOVE, list(map(int, box))))

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return PascalVOCDataset.CLASSES[class_id]
