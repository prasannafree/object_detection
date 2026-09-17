import os
import sys
import torch
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from datapipeline.coco_dataset import COCODataset

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = tensor.permute(1, 2, 0).numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1) * 255
    return img.astype(np.uint8)

def test_dataset():
    print("Initializing dataset...")
    # Using val dataset for faster loading, but with training augmentations to test them
    dataset = COCODataset(json_file=cfg.val_json, img_dir=cfg.val_img_dir, img_size=cfg.img_size, is_train=True)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a random image
    img, boxes, labels = dataset[42]
    
    print(f"Image tensor shape: {img.shape}")
    print(f"Boxes tensor shape: {boxes.shape}")
    print(f"Labels tensor shape: {labels.shape}")
    
    # Assertions
    assert img.shape == (3, cfg.img_size, cfg.img_size)
    assert boxes.shape[1] == 4
    assert len(boxes) == len(labels)
    assert (boxes >= 0).all() and (boxes <= cfg.img_size).all()
    assert (labels >= 0).all() and (labels < cfg.num_classes).all()
    
    # Visualize
    img_viz = denormalize(img)
    img_viz = cv2.cvtColor(img_viz, cv2.COLOR_RGB2BGR)
    
    boxes = boxes.numpy()
    labels = labels.numpy()
    
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_viz, f"Class {label}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    os.makedirs('outputs', exist_ok=True)
    out_path = 'outputs/dataset_test.jpg'
    cv2.imwrite(out_path, img_viz)
    print(f"Test visualization saved to {out_path}")
    print("Dataset test passed successfully!")

if __name__ == "__main__":
    test_dataset()
