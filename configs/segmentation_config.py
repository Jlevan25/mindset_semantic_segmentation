import os
import time


class MainConfig:
    def __init__(self):
        self.epochs = 1000
        self.seed = 1


class TrainerConfig:
    def __init__(self):
        dataset = 'DUTS'
        # dataset = 'COCO'

        self.ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATASET_PATH = os.path.join(self.ROOT, 'datasets', dataset)

        self.model_params = ModelConfig()
        self.device = 'cpu'

        self.batch_size = 1
        self.lr = 0.001

        self.debug = True
        self.show_each = 3

        # if need overfit (batch_size, shuffle=False) else None, True
        self.overfit_on_batch = None
        self.shuffle = True

        postfix = '_overfitted_on_batch' if self.overfit_on_batch is not None else ''

        experiment_name = f'model_{self.model_params.model_name}_batch_size{self.batch_size}_lr_{self.lr}_{time.time()}'
        self.LOG_PATH = os.path.join(self.ROOT, 'logs', dataset, experiment_name + postfix)

        self.SAVE_PATH = os.path.join(self.ROOT, 'checkpoints', dataset,
                                      f'lr{self.lr}_start_filters{self.model_params.start_filters}_{time.time()}' +
                                      postfix)

        self.LOAD_PATH = os.path.join(self.ROOT, 'checkpoints', dataset,
                                      'lr0.001_start_filters32_1651247892.1799445_overfitted_on_batch', '237.pth')

        self.mask_stride = self.model_params.output_shape

        self.mirror_padding = (self.model_params.input_shape[0] - self.model_params.output_shape[0]) // 2, \
                              (self.model_params.input_shape[1] - self.model_params.output_shape[1]) // 2


class ModelConfig:
    def __init__(self):
        self.model_name = 'Unet'
        self.in_channels = 3
        self.out_channels = 2
        self.start_filters = 32

        self.input_shape = 348, 348
        self.output_shape = 164, 164
