from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Dataset paths
    data_dir: str = "data"
    train_json: str = "data/annotations/instances_train2017.json"
    val_json: str = "data/annotations/instances_val2017.json"
    train_img_dir: str = "data/train2017"
    val_img_dir: str = "data/val2017"
    
    # Image properties
    img_size: int = 640
    num_classes: int = 80
    
    # FPN & Head architecture
    fpn_channels: int = 128
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])
    
    # Training hyperparams
    batch_size: int = 16
    lr: float = 0.01
    weight_decay: float = 1e-4
    epochs: int = 50
    
    # Inference thresholds
    score_threshold: float = 0.05
    nms_threshold: float = 0.5

# Global instance
cfg = Config()
