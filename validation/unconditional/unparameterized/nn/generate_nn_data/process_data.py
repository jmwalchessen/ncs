import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CustomSpatialImageAndClassDataset(Dataset):
    def __init__(self, images, classes):
        self.images = images
        self.classes = classes

    def __len__(self):
        return ((self.images).shape[0])

    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        class_label = self.classes[idx]
        return image, class_label
    
