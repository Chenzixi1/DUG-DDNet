import torch
from torchvision.transforms import functional as F


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, depth=None):
        for t in self.transforms:
            image, target, depth = t(image, target, depth)
        return image, target, depth

class ToTensor():
    def __call__(self, image, target, depth=None):
        image = F.to_tensor(image)
        depth = F.to_tensor(depth) if depth is not None else None

        return image, target, depth

class Normalize():
    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr
        self.mean_dep = [4413.160626995486]  # DORN
        self.std_dep = [3270.0158918863494]


    def __call__(self, image, target, depth=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        depth = F.normalize(depth, mean=self.mean_dep, std=self.std_dep) if depth is not None else None

        if self.to_bgr:
            image = image[[2, 1, 0]]
        
        return image, target, depth
