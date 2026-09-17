
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

class COCODataset(Dataset):
    def __init__(self, json_file, img_dir, img_size=640, is_train=True):
        """
        Args:
            json_file: Path to COCO annotations JSON.
            img_dir: Directory with COCO images.
            img_size: Target image size (will be resized and padded).
            is_train: If True, apply training augmentations.
        """
        self.img_dir = img_dir
        self.coco = COCO(json_file)
        self.img_size = img_size
        self.is_train = is_train

        # Filter out images without annotations
        self.img_ids = []
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            if len(ann_ids) > 0:
                self.img_ids.append(img_id)

        # COCO category IDs are not contiguous 1-80 (they go up to 90)
        # We need to map them to 0-79
        self.coco_cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_idx = {cat['id']: i for i, cat in enumerate(self.coco_cats)}
        
        # Setup albumentations transforms
        self.transforms = self._get_transforms()

    def _get_transforms(self):
        # Base transforms: resize to square, normalize, and convert to tensor
        transforms_list = [
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, 
                          border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114)),
        ]
        
        if self.is_train:
            transforms_list.extend([
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ])
            
        transforms_list.extend([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # Bbox format 'pascal_voc' is [x_min, y_min, x_max, y_max]
        return A.Compose(transforms_list, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        
        for ann in anns:
            # COCO bbox format is [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # Filter out invalid boxes
            if w <= 0 or h <= 0:
                continue
                
            # Convert to [x_min, y_min, x_max, y_max]
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            
            boxes.append([x_min, y_min, x_max, y_max])
            # Map original category ID to 0-79 index
            labels.append(self.cat_id_to_idx[ann['category_id']])
            
        if len(boxes) == 0:
            # Fallback if all boxes were invalid (unlikely but possible)
            return self.__getitem__((idx + 1) % len(self))
            
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply albumentations
        transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
        
        image_t = transformed['image']
        boxes_t = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        labels_t = torch.tensor(transformed['labels'], dtype=torch.int64)
        
        # If augments drop all boxes (e.g. crop), try again
        if len(boxes_t) == 0:
             return self.__getitem__((idx + 1) % len(self))
             
        return image_t, boxes_t, labels_t

def collate_fn(batch):
    """
    Since images have different numbers of bounding boxes, we cannot stack them 
    into a single tensor directly. This custom collate function groups them as lists.
    """
    images = []
    targets = []
    
    for image, boxes, labels in batch:
        images.append(image)
        targets.append({
            'boxes': boxes,
            'labels': labels
        })
        
    images = torch.stack(images, dim=0)
    return images, targets
