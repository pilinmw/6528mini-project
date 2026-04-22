import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class SingleImageDataset(Dataset):
    def __init__(self, image_path, target_height=None):
        """
        Loads an image, optionally resizes it, and prepares pixel coordinates and colors.
        Args:
            image_path (str): Path to the image.
            target_height (int): If provided, resizes the image preserving aspect ratio.
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if requested
        if target_height is not None:
            h, w = img.shape[:2]
            aspect_ratio = w / h
            target_width = int(target_height * aspect_ratio)
            img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
        self.H, self.W = img.shape[:2]
        
        # Normalize colors to [0, 1]
        self.img_rgb = torch.tensor(img, dtype=torch.float32) / 255.0
        
        # Create coordinates meshgrid
        # Start from -1 to 1 to help neural network optimization
        y_coords = torch.linspace(-1, 1, self.H)
        x_coords = torch.linspace(-1, 1, self.W)
        
        # Matrix of shape [H, W, 2]
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Flatten to [H*W, 2]
        self.coords = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
        
        # Flatten colors to [H*W, 3]
        self.colors = self.img_rgb.view(-1, 3)

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        return self.coords[idx], self.colors[idx]

if __name__ == "__main__":
    # Test dataset
    import sys
    # For testing, assume jcsmr-1.jpg is in the parent directory
    ds = SingleImageDataset("../jcsmr-1.jpg", target_height=256)
    print(f"Dataset length: {len(ds)}")
    print(f"Original shape: {ds.H}x{ds.W}")
    print(f"First coord: {ds[0][0]}, First color: {ds[0][1]}")
    print(f"Last coord: {ds[-1][0]}, Last color: {ds[-1][1]}")
