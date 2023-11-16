import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
import cv2


class Scans(Dataset):
    """
    num_classes: 2
    """

    def __init__(self, root: str, mode: str = 'train', transform=None) -> None:
        super().__init__()
        self.CLASSES = ['background', 'object']
        self.IMAGE_SUBSET = 'hard_images'

        self.PALETTE = torch.tensor(
            [[255, 255, 255], [0, 0, 0]])

        self.label_mapping = {0: 0,
                              255: 1}

        assert mode in ['train', 'validation', 'test']
        self.transform = None  # transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 100

        images = Path(root).joinpath("hard_images").joinpath(mode).joinpath("images")
        labels = Path(root).joinpath("hard_images").joinpath(mode).joinpath("annotations")
        self.images = sorted(list(images.rglob('*.tif')))
        self.labels = sorted(list(labels.rglob('*.png')))

        if self.images == 0 or self.labels == 0:
            raise Exception(f"No images found in dataset {root}")
        print(f"Found {len(self.images)} dataset examples and {len(self.labels)} corresponding labels")

    def __len__(self) -> int:
        return min(len(self.images), len(self.labels))

    def encode(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return torch.from_numpy(label)

    def compare_ids(self, image_path, annotation_path):
        image_id = image_path.split('/')[-1].split('_')[0]
        annotation_id = annotation_path.split('/')[-1].split('_')[0]

        return image_id == annotation_id

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.images[index])
        lbl_path = str(self.labels[index])
        assert self.compare_ids(img_path, lbl_path)

        # read example
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        height, width = image.shape
        new_width = 2048
        new_height = 1024
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = (width + new_width) // 2
        bottom = (height + new_height) // 2

        image = image[top:bottom, left:right]

        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image = np.transpose(image, (2, 0, 1))  # coping because we have gray image, but NN input is 3-ch

        label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        label = label[top:bottom, left:right]
        if self.transform:
            image, label = self.transform(image, label)
        return image, self.encode(label.squeeze()).long()
