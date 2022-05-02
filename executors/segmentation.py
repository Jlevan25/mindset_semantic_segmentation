import os.path

import cv2
import numpy as np
import math
import torch
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau

import models
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import functional as transforms_F
from transforms import MirrorPad, MosaicSplit, ZeroPad
from datasets import CocoSegmentationDataset, DUTSDataset
from configs import UnetConfig, TrainerConfig, MainConfig
from utils import one_hot_argmax
from metrics import MeanIoU, MeanPixelAccuracy
from losses import SegmentationCrossEntropy
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, writer, cfg: dict, transform: dict = None, target_transform: dict = None, metrics: list = None):
        self.cfg = cfg['trainer']
        self.model = models.UNet(in_channels=self.cfg.model_params.in_channels,
                                 out_channels=self.cfg.model_params.out_channels,
                                 start_filters=self.cfg.model_params.start_filters,
                                 config=cfg['net'])

        if self.cfg.device == 'cuda':
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, **self.cfg.scheduler_kwargs)
        self.criterion = SegmentationCrossEntropy()
        self.metrics = metrics
        self.transform = transform
        self.target_transform = target_transform
        self.device = self.cfg.device
        self.writer = writer

        self.datasets, self.dataloaders = dict(), dict()
        self._global_step = dict()

    def _get_data(self, data_type):
        self._global_step[data_type] = -1
        transform = self.transform[data_type] if self.transform is not None else None
        target_transform = self.target_transform[data_type] if self.target_transform is not None else None

        # self.datasets[data_type] = CocoSegmentationDataset(root=os.path.join(self.cfg.DATASET_PATH, data_type),
        #                                                    annFile=os.path.join(self.cfg.DATASET_PATH, 'annotations',
        #                                                                         f'instances_{data_type}.json'),
        #                                                    transform=transform,
        #                                                    target_transform=target_transform,
        #                                                    overfit_batch_size=self.cfg.overfit_on_batch)

        self.datasets[data_type] = DUTSDataset(root=os.path.join(self.cfg.DATASET_PATH, data_type, 'images'),
                                               target_root=os.path.join(self.cfg.DATASET_PATH, data_type, 'masks'),
                                               transform=transform,
                                               target_transform=target_transform,
                                               overfit_batch_size=self.cfg.overfit_on_batch)

        def collate_fn(data):
            img, target = data[0][0], data[0][1]
            for d in data[1:]:
                img, target = torch.cat((img, d[0]), dim=0), torch.cat((target, d[1]), dim=0)
            return img, target

        self.dataloaders[data_type] = DataLoader(self.datasets[data_type], batch_size=self.cfg.batch_size,
                                                 collate_fn=collate_fn, shuffle=self.cfg.shuffle)

    @torch.no_grad()
    def _calc_epoch_metrics(self, stage):
        self._calc_metrics(stage, self.cfg.debug, is_epoch=True)

    @torch.no_grad()
    def _calc_batch_metrics(self, masks, targets, stage, debug):
        self._calc_metrics(stage, debug, one_hot_argmax(masks), targets)

    def _calc_metrics(self, stage, debug, *batch, is_epoch: bool = False):
        for metric in self.metrics:
            values = metric(is_epoch, *batch).tolist()
            metric_name = type(metric).__name__

            for cls, scalar in (zip(self.classes, values) if hasattr(self, 'classes') else enumerate(values)):
                self.writer.add_scalar(f'{stage}/{metric_name}/{cls}', scalar, self._global_step[stage])

            self.writer.add_scalar(f'{stage}/{metric_name}/overall',
                                   sum(values) / len(values), self._global_step[stage])

            if debug:
                print("{}: {}".format(metric_name, values))

    def _epoch_step(self, stage='test', epoch=None):

        if stage not in self.dataloaders:
            self._get_data(stage)

        calc_metrics = self.metrics is not None and self.metrics
        print('\n_______', stage, f'epoch{epoch}' if epoch is not None else '',
              'len:', len(self.dataloaders[stage]), '_______')

        for i, (images, targets) in enumerate(self.dataloaders[stage]):

            self._global_step[stage] += 1
            debug = self.cfg.debug and i % self.cfg.show_each == 0

            if debug:
                print('\n___', f'Iteration {i}', '___')

            one_hots = F.one_hot(targets.round().long(),
                                 num_classes=self.cfg.model_params.out_channels).transpose(1, -1).squeeze(-1)

            # for i, kernel in enumerate(one_hots):
            #     cv2.imshow(f'target{i}', kernel[1, None].permute(1, 2, 0).numpy().astype('uint8') * 255)
            # cv2.waitKey()

            masks = self.model(images.to(self.device))

            if calc_metrics:
                self._calc_batch_metrics(masks, one_hots, stage, debug)

            if stage == 'train':
                loss = self.criterion(masks, one_hots.float())
                self.writer.add_scalar(f'{stage}/loss', loss, self._global_step[stage])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss.detach())

                if debug:
                    print(f'Train Loss: {loss.item()}')

        if calc_metrics and epoch is not None:
            print('\n___', f'Epoch Summary', '___')
            self._calc_epoch_metrics(stage)

    def fit(self, i_epoch):
        self._epoch_step(stage='train', epoch=i_epoch)

    @torch.no_grad()
    def validation(self, i_epoch):
        self._epoch_step(stage='val', epoch=i_epoch)

    @torch.no_grad()
    def test(self):
        self._epoch_step(stage='test')

    @torch.no_grad()
    def get_masks(self, images):
        images = images if isinstance(images, list) else [images]
        masks = [self.get_mask(img) for img in images]

        return masks

    @torch.no_grad()
    def get_mask(self, img):
        kernels = self.transform['test'](img)
        img_shape = img.size[::-1]

        h = math.ceil(img_shape[0] / self.cfg.model_params.output_shape[0])
        w = math.ceil(img_shape[1] / self.cfg.model_params.output_shape[0])

        kernels = kernels.reshape(h, w, *kernels.shape[1:])
        out_shape = self.cfg.model_params.out_channels, self.cfg.model_params.output_shape[1]
        masks = [self.model(kernel.to(self.device)).permute(1, 2, 0, 3).reshape((*out_shape, -1)) for kernel in kernels]

        mask = transforms_F.center_crop(torch.cat(masks, dim=1), img_shape)
        return mask.argmax(dim=0)

    def save_model(self, epoch):
        path = os.path.join(self.cfg.SAVE_PATH, f'{epoch}.pth')

        if not os.path.exists(self.cfg.SAVE_PATH):
            os.makedirs(self.cfg.SAVE_PATH)

        torch.save(self.model.state_dict(), path)
        print('model saved')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print('model loaded')


def main():
    trainer_cfg = TrainerConfig()
    net_cfg = UnetConfig()
    cfg = MainConfig()

    configs = dict(trainer=trainer_cfg, net=net_cfg)

    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)

    unet_preprocess_block = [MirrorPad(trainer_cfg.mirror_padding,
                                       trainer_cfg.model_params.output_shape),
                             transforms.ToTensor(),
                             # todo: get mu, sigma from all channels
                             # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             #                      std=[0.229, 0.224, 0.225]),

                             # DUTS
                             transforms.Normalize(mean=[0.4924, 0.4633, 0.3969],
                                                  std=[0.2264, 0.2237, 0.2289]),
                             MosaicSplit(kernel_size=trainer_cfg.model_params.input_shape,
                                         in_depth=trainer_cfg.model_params.in_channels,
                                         stride=trainer_cfg.mask_stride)]

    train_preprocess = transforms.Compose([*unet_preprocess_block])

    eval_preprocess = transforms.Compose(unet_preprocess_block)

    target_preprocess = transforms.Compose([ZeroPad(kernel_size=trainer_cfg.model_params.output_shape),
                                            transforms.ToTensor(),
                                            MosaicSplit(kernel_size=trainer_cfg.model_params.output_shape,
                                                        stride=trainer_cfg.model_params.output_shape)])

    metrics = [MeanPixelAccuracy(trainer_cfg.model_params.out_channels), MeanIoU(trainer_cfg.model_params.out_channels)]

    writer = SummaryWriter(log_dir=trainer_cfg.LOG_PATH)
    trainer = Trainer(writer, configs,
                      metrics=metrics,
                      transform=dict(train=train_preprocess, val=eval_preprocess, test=eval_preprocess),
                      target_transform=dict(train=target_preprocess, val=target_preprocess, test=target_preprocess))

    # get mask from image
    img_path = os.path.join(trainer_cfg.DATASET_PATH, 'train', 'images')
    target_path = os.path.join(trainer_cfg.DATASET_PATH, 'train', 'masks')
    trainer.load_model(trainer_cfg.LOAD_PATH)
    show_img = None
    for i in range(6):
        masks = trainer.get_masks(Image.open(os.path.join(img_path, os.listdir(img_path)[i])).convert("RGB"))

        img = cv2.imread(os.path.join(img_path, os.listdir(img_path)[i]))
        target = cv2.imread(os.path.join(target_path, os.listdir(target_path)[i]))
        concat_img = np.concatenate((img, target,
                                     masks[0].numpy()[..., None].repeat(3, axis=-1).astype('uint8') * 255), axis=1)
        cv2.imwrite(f'img_target_prediction{i}.jpg', concat_img)

    # trainer.load_model(trainer_cfg.LOAD_PATH)
    # for epoch in range(cfg.epochs):
    #     trainer.fit(epoch)
    #     trainer.save_model(epoch)
    #
    #     trainer.writer.add_scalar(f'scheduler lr', trainer.optimizer.param_groups[0]['lr'], epoch)
    #     # trainer.validation(epoch)
    #
    # trainer.test()


if __name__ == '__main__':
    main()
