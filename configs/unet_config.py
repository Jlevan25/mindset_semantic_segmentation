class UnetConfig:
    def __init__(self):
        self.n_blocks = 5

        self.conv_mode = 'valid'
        self.conv_kwargs = dict(kernel_size=3, stride=1, bias=True)
        self.conv_transpose_kwargs = dict(kernel_size=2, stride=2)

        self.activation = 'relu'
        self.activation_kwargs = dict(inplace=True)

        self.pooling = 'max'
        self.pooling_kwargs = dict(kernel_size=2, stride=2, padding=0)

        self.normalization = None
