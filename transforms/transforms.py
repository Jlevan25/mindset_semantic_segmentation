import torch
from torch import Tensor
from torchvision.transforms import Pad, functional as F
from utils import get_kernel_indexes3d


class _Flex_Pad(Pad):
    def __init__(self, padding, kernel_size, fill, padding_mode):
        super().__init__(padding, fill, padding_mode)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Padded image.
        """
        img_size = img.shape[-2:] if torch.is_tensor(img) else img.size[::-1]

        img_shape = torch.tensor(img_size)
        kernel_size = torch.tensor(self.kernel_size)

        padding = kernel_size * (img_shape / kernel_size).ceil() - img_shape
        padding = torch.ceil(padding / 2)
        padding = padding.int() + torch.tensor(self.padding).reshape(-1, 2)
        return F.pad(img, padding.reshape(-1).tolist()[::-1], self.fill, self.padding_mode)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MirrorPad(_Flex_Pad):
    def __init__(self, padding=0, kernel_size=1, fill=0):
        super().__init__(padding, kernel_size, fill, 'reflect')


class ZeroPad(_Flex_Pad):
    def __init__(self, padding=0, kernel_size=1, fill=0):
        super().__init__(padding, kernel_size, fill, 'constant')


class MosaicSplit(torch.nn.Module):
    def __init__(self, kernel_size, in_depth=1, stride=1):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.in_depth = in_depth
        self._indexes = dict()

    def forward(self, img):
        """
        Args:
            img (Tensor): Image to be Mosaic Splited.

        Returns:
            Tensor: Mosaic image.
        """

        h, w = img.shape[-2:]
        output_shape = (h - self.kernel_size[0]) // self.stride[0] + 1, \
                       (w - self.kernel_size[1]) // self.stride[1] + 1

        k, i, j = get_kernel_indexes3d((self.in_depth, *self.kernel_size), output_shape, self.stride)
        return img[..., k, i, j].reshape(-1, self.in_depth, *self.kernel_size)
