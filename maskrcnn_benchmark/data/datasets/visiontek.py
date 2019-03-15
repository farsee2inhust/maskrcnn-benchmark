import torch
import torchvision
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from PIL import Image
import os
import json

class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def encode(Str):
    myList = []
    list_float = list(map(float, Str.strip().split()))
    X = list_float[0::2]
    Y = list_float[1::2]
    assert len(X) == len(Y)
    for i in range(len(X)):
        if (abs(X[i] - X[i - 1]) > 1e-5 or abs(Y[i] - Y[i - 1]) > 1e-5): # 去掉连续重复的点
            myList.append(Vertex(Y[i], X[i]))   # 这里xy对换
    return myList

class VisiontekDataset(object):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.imgList = os.listdir(img_dir)
        self.transforms = transforms
        with open(os.path.join(ann_dir, "size.json"), "r") as size_f:
            self.size = json.load(size_f)


    def __getitem__(self, idx):
        # load the image as a PIL Image
        img_name = self.imgList[idx]
        ann_name = img_name.replace("jpg", "txt")
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        boxes = []
        masks = []
        with open(os.path.join(self.ann_dir, ann_name), "r") as ann_f:
            for ann_str in ann_f:
                if len(ann_str.strip()) == 0:
                    continue
                v_list = encode(ann_str)
                # 构造bbox
                min_x = v_list[0].x
                max_x = v_list[0].x
                min_y = v_list[0].y
                max_y = v_list[0].y
                for v in v_list:
                    if v.x < min_x:
                        min_x = v.x
                    if v.x > max_x:
                        max_x = v.x
                    if v.y < min_y:
                        min_y = v.y
                    if v.y > max_y:
                        max_y = v.y
                box = [min_x, min_y, max_x, max_y]
                boxes.append(box)
                # 构造mask
                mask = []
                for v in v_list:
                    mask.append(v.x)
                    mask.append(v.y)
                masks.append([mask])
        # 处理bbox
        if len(boxes) == 0:
            print(ann_name)
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, img.size, mode="xyxy")
        # 处理标签
        classes = [1 for i in range(len(boxes))]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        # 处理mask
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)
        # TODO 不知道这一步有什么用
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)


        return img, target, idx


    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        img_name = self.imgList[idx]
        return {"height": self.size[img_name]["height"], "width": self.size[img_name]["width"]}

    def __len__(self):
        return len(self.imgList)