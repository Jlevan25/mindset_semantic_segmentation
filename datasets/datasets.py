from typing import Optional, Callable, List, Tuple, Any

import torch
from PIL import Image
from torchvision.datasets import CocoDetection, VisionDataset

from os import listdir
from os.path import isfile, join

import numpy as np


class CocoSegmentationDataset(CocoDetection):

    def __init__(self, root: str, annFile: str,
                 transform=None, target_transform=None, transforms=None, overfit_batch_size: int = None):
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.overfit_batch_size = overfit_batch_size

    def _load_target(self, id: int):
        target = super()._load_target(id)
        return self._anns2mask(target)

    def _anns2mask(self, anns):
        if not anns:
            return None

        masks = np.array([self.coco.annToMask(ann) for ann in anns])

        # todo: for multiple class mul class_id

        # masks = [torch.from_numpy(self.coco.annToMask(ann) * ann['category_id'])
        #          for ann in anns]

        mask = np.amax(masks, axis=0) * 255

        return Image.fromarray(mask)

    def __len__(self) -> int:
        return len(self.ids) if self.overfit_batch_size is None else self.overfit_batch_size


class DUTSDataset(VisionDataset):

    def __init__(
            self,
            root: str,
            target_root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            overfit_batch_size: int = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.target_root = target_root
        self.samples = [f.split('.')[0] for f in listdir(root) if isfile(join(root, f))]
        self.overfit_batch_size = overfit_batch_size

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_path = self.samples[index]
        image = Image.open(join(self.root, f'{image_path}.jpg')).convert("RGB")
        target = Image.open(join(self.target_root, f'{image_path}.png'))

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return self.overfit_batch_size if self.overfit_batch_size is not None else len(self.samples)
