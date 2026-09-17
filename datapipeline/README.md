# Data Pipeline

This module handles loading the COCO dataset, parsing bounding boxes, applying augmentations, and preparing PyTorch tensors for our object detection model.

## Overview

The `COCODataset` class (in `coco_dataset.py`) inherits from PyTorch's `Dataset`. Its primary job is to take an image ID, read the image from disk, extract all ground-truth bounding boxes, augment them, and return them as a neat bundle.

## Step-by-Step Flow

When the DataLoader requests an image (e.g., `dataset[0]`), the following happens:

1. **Load Image & Annotations:**
   - The image is loaded using OpenCV (as BGR) and converted to RGB.
   - The corresponding annotations (bounding boxes and categories) are fetched using `pycocotools`.

2. **Format Conversion:**
   - **COCO Format:** `[x_top_left, y_top_left, width, height]`
   - **Pascal VOC Format (Our target):** `[x_min, y_min, x_max, y_max]`
   - The pipeline iterates through all boxes for the image and converts them.
   - Category IDs are remapped from COCO's sparse 1-90 range to a contiguous 0-79 index.

3. **Augmentations (`albumentations`):**
   - The image is resized so its longest side is exactly `img_size` (e.g., 640).
   - If the image is not square (e.g., 640x480), gray padding is added to make it 640x640. 
   - **Crucially:** Albumentations automatically scales and shifts the bounding boxes to match the resized and padded image!
   - (Training only) Random horizontal flips and color jittering are applied.
   - The image is normalized using ImageNet mean/std and converted to a PyTorch tensor.

4. **Collate Function (`collate_fn`):**
   - In classification tasks, `torch.stack` can combine a batch of images and labels into a single tensor.
   - In object detection, Image A might have 2 boxes, while Image B has 15 boxes. PyTorch cannot stack tensors of different sizes.
   - The `collate_fn` intercepts the batch creation. It stacks the images into a `(B, 3, 640, 640)` tensor, but keeps the bounding boxes and labels as a **list of dictionaries** (one dictionary per image).

## Example Output Structure

A single batch returned by the DataLoader will look like this:

```python
images, targets = next(iter(dataloader))

# images: Tensor of shape (Batch_Size, 3, 640, 640)
# targets: List of length Batch_Size. 

# Example of targets[0] (The annotations for the first image in the batch):
{
    'boxes': tensor([[10.5, 20.1, 100.0, 200.5],    # Box 1 [x1, y1, x2, y2]
                     [45.0, 50.0, 80.0, 90.0]]),    # Box 2 [x1, y1, x2, y2]
                     
    'labels': tensor([4, 1])                        # Class indices (e.g., Airplane, Bicycle)
}
```
