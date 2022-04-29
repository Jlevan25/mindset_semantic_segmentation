from typing import Type

from configs import UnetConfig
from torch import nn
from torchvision.transforms import functional as transforms_F
import torch
from collections import deque


# @torch.jit.script
# def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
#     """
#     Center-crops the encoder_layer to the size of the decoder_layer,
#     so that merging (concatenation) between levels/blocks is possible.
#     This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
#     """
#     if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
#         ds = encoder_layer.shape[2:]
#         es = decoder_layer.shape[2:]
#         assert ds[0] >= es[0]
#         assert ds[1] >= es[1]
#         if encoder_layer.dim() == 4:  # 2D
#             encoder_layer = encoder_layer[
#                             :,
#                             :,
#                             ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
#                             ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
#                             ]
#         elif encoder_layer.dim() == 5:  # 3D
#             assert ds[2] >= es[2]
#             encoder_layer = encoder_layer[
#                             :,
#                             :,
#                             ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
#                             ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2),
#                             ((ds[2] - es[2]) // 2):((ds[2] + es[2]) // 2),
#                             ]
#     return encoder_layer, decoder_layer


class DownBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation_type: Type,
                 config: UnetConfig,
                 padding: tuple,
                 pooling_type: Type = None,
                 normalization_type: Type = None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # conv layers
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, padding=padding, **config.conv_kwargs)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, padding=padding, **config.conv_kwargs)

        self.activation = activation_type(**config.activation_kwargs)
        self.pooling = pooling_type(**config.pooling_kwargs) if pooling_type is not None else None
        self.normalization = True if normalization_type is not None else False
        # self.pooling = pooling
        # self.activation = activation

        # normalization layers
        if self.normalization:
            self.norm1 = normalization_type(self.out_channels)
            self.norm2 = normalization_type(self.out_channels)

    def forward(self, x):
        y = self.conv1(x)  # convolution 1
        y = self.activation(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1

        y = self.conv2(y)  # convolution 2
        y = self.activation(y)  # activation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2

        before_pooling = y  # save the outputs before the pooling operation
        if self.pooling is not None:
            y = self.pooling(y)  # pooling
        return y, before_pooling


class UpBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation_type: Type,
                 config: UnetConfig,
                 padding: tuple,
                 normalization_type: Type = None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # upconvolution/upsample layer
        self.up_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, **config.conv_transpose_kwargs)

        # conv layers
        self.conv1 = nn.Conv2d(2 * self.out_channels, self.out_channels, padding=padding, **config.conv_kwargs)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, padding=padding, **config.conv_kwargs)

        # activation layers
        self.activation = activation_type(**config.activation_kwargs)
        # self.act1 = get_activation(self.activation)
        # self.act2 = get_activation(self.activation)

        self.normalization = True if normalization_type is not None else False

        # normalization layers
        if self.normalization:
            self.norm0 = normalization_type(self.out_channels)
            self.norm1 = normalization_type(self.out_channels)
            self.norm2 = normalization_type(self.out_channels)

    def forward(self, saved_tensor, input_tensor):

        y = self.up_conv(input_tensor)  # up-convolution
        cropped_saved_tensor = transforms_F.center_crop(saved_tensor, y.shape[-2:])
        # cropped_encoder_layer, dec_layer = autocrop(encoder_layer, up_layer)  # cropping

        y = self.activation(y)  # activation 0
        if self.normalization:
            y = self.norm0(y)  # normalization 0

        merged_layer = torch.cat((y, cropped_saved_tensor), dim=1)  # concatenation
        y = self.conv1(merged_layer)  # convolution 1
        y = self.activation(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1

        y = self.conv2(y)  # convolution 2
        y = self.activation(y)  # acivation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2
        return y


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 start_filters: int,
                 config: UnetConfig):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = config.n_blocks

        activation = self._get_activation(config.activation)
        normalization = self._get_normalization(config.normalization)
        pooling = self._get_pooling(config.pooling)
        padding = self._get_padding(config.conv_mode, config.conv_kwargs['kernel_size'])

        self.down_blocks = []
        self.up_blocks = []

        # create encoder path
        num_filters_out = 0
        for i in range(self.n_blocks):
            num_filters_in = self.in_channels if i == 0 else num_filters_out
            num_filters_out = start_filters * (2 ** i)
            pool = pooling if i < self.n_blocks - 1 else None

            down_block = DownBlock(in_channels=num_filters_in,
                                   out_channels=num_filters_out,
                                   padding=padding,
                                   pooling_type=pool,
                                   activation_type=activation,
                                   normalization_type=normalization,
                                   config=config)

            self.down_blocks.append(down_block)

        # create decoder path (requires only n_blocks-1 blocks)
        for i in range(self.n_blocks - 1):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            up_block = UpBlock(in_channels=num_filters_in,
                               out_channels=num_filters_out,
                               padding=padding,
                               activation_type=activation,
                               normalization_type=normalization,
                               config=config)

            self.up_blocks.append(up_block)

        # final convolution
        self.conv_final = nn.Conv2d(num_filters_out, self.out_channels, kernel_size=1, stride=1, padding=0,
                                    bias=True)

        # add the list of modules to current module
        # self.down_blocks = nn.ModuleList(self.down_blocks)
        # self.up_blocks = nn.ModuleList(self.up_blocks)

        self.down_blocks = nn.Sequential(*self.down_blocks)
        self.up_blocks = nn.Sequential(*self.up_blocks)

        # initialize the weights
        self.initialize_parameters()

    @staticmethod
    def _get_activation(activation: str):
        if activation == 'relu':
            return nn.ReLU
        elif activation == 'leaky':
            return nn.LeakyReLU
        elif activation == 'elu':
            return nn.ELU
        else:
            raise ValueError(f'relu, leaky or elu activation expected, but {activation}')

    @staticmethod
    def _get_normalization(normalization: str = None):
        if normalization is None:
            return None

        if normalization == 'batch':
            return nn.BatchNorm2d
        elif normalization == 'instance':
            return nn.InstanceNorm2d
        else:
            raise ValueError(f'batch or instance normalization expected, but {normalization}')

    @staticmethod
    def _get_pooling(pooling: str = None):
        if pooling is None:
            return None

        if pooling == 'max':
            return nn.MaxPool2d
        elif pooling in ['avg', 'mean']:
            return nn.AvgPool2d
        else:
            raise ValueError(f'max or avg pooling expected, but {pooling}')

    @staticmethod
    def _get_padding(conv_mode: str, kernel_size=None):
        if conv_mode == 'same' and kernel_size is not None:
            kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
            return (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2
        elif conv_mode == 'valid':
            return 0, 0
        else:
            raise ValueError(f'same with kernel_size or valid conv_mode expected, but {conv_mode}')

    def initialize_parameters(self,
                              weights_init_def=nn.init.xavier_uniform_,
                              bias_init_def=nn.init.zeros_,
                              weights_kwargs=None,
                              bias_kwargs=None):

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                weights_init_def(module.weight, **weights_kwargs) if weights_kwargs is not None \
                    else weights_init_def(module.weight)
                if module.bias is not None:
                    bias_init_def(module.bias, **bias_kwargs) if bias_kwargs is not None \
                        else weights_init_def(module.weight)

    def forward(self, x: torch.tensor):
        down_block_output = deque()

        # Encoder pathway
        for module in self.down_blocks:
            x, before_pooling = module(x)
            down_block_output.append(before_pooling)

        down_block_output.pop()

        # Decoder pathway
        for i, module in enumerate(self.up_blocks):
            before_pool = down_block_output.pop()
            x = module(before_pool, x)

        x = self.conv_final(x)

        return x

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if
                      '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'


if __name__ == '__main__':
    from torchsummary import summary

    cfg = UnetConfig()
    model = UNet(in_channels=3,
                 out_channels=2,
                 start_filters=64,
                 config=cfg)
    summary = summary(model, (3, 348, 348), batch_size=32)
    print(model)
