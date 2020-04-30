from collections import OrderedDict
import torch
import numpy as np
import logging

from tracknet.core._layerutils import *
from tracknet.core.layers import *

class TrackNet(nn.Module):
    def __init__(self, image_shape=(360, 640, 3), seq_num=3, batch_norm=True):
        """
        :param image_shape: array-like, (height, width, channel)
        :param seq_num: int, how many images will be input to model
        :param batch_norm: bool, whether to append batch normalization layer
        """
        super().__init__()

        assert len(image_shape) == 3, "image must be 3d"
        self.image_shape = image_shape
        self.seq_num = seq_num
        self._batch_norm = batch_norm

        # set batch norm flag
        Conv2dRelu.batch_norm = self._batch_norm
        Deconv2dRelu.batch_norm = self._batch_norm

        vgg_layers = [
            *Conv2dRelu.block('1', 2, self.input_channel, 64),

            *Conv2dRelu.block('2', 2, 64, 128),

            *Conv2dRelu.block('3', 3, 128, 256),

            *Conv2dRelu.block('4', 3, 256, 512)
        ]
        vgg_layers = vgg_layers[:-1] # remove last maxpooling layer

        deconvnet_layers = [
            *Deconv2dRelu.block('5', 3, 512, 256),

            *Deconv2dRelu.block('6', 2, 256, 128),

            *Deconv2dRelu.block('7', 2, 128, 64)
        ]

        self.tracknet_layers = nn.ModuleDict(OrderedDict(vgg_layers + deconvnet_layers))

        feature_layers = [
            *Conv2dRelu.one('8', 64, 256, kernel_size=(3, 3), padding=1),

            ('softmax', SoftMax())#nn.Softmax(dim=1)) # dim = 1 means along with channel of (b, c, h, w)
        ]
        self.feature_layers = nn.ModuleDict(feature_layers)

        self.heatmap = HeatMap()
        self.detector = Detector(threshold=128)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    @property
    def input_height(self):
        return self.image_shape[0]
    @property
    def input_width(self):
        return self.image_shape[1]
    @property
    def input_channel(self):
        return self.image_shape[2] * self.seq_num

    def forward(self, x):
        for name, layer in self.tracknet_layers.items():
            x = layer(x)

        for name, layer in self.feature_layers.items():
            x = layer(x)

        if self.training:
            return x # shape = (h, w, c)
        else:
            # x's shape = (h, w)
            heatmaps = self.heatmap(x) # (b, c, h, w)-> (b, h, w)
            return heatmaps

    def inference(self, imgs):
        """
        :param imgs: list of img(ndarray or Tensor), Tensor or ndarray. Note that if it's Tensor or ndarray, shape must be (c, h, w) or (1, c, h, w)
        :return:
        """
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        elif isinstance(imgs, torch.Tensor):
            pass
        elif isinstance(imgs, (list, tuple)):
            if all([isinstance(img, np.ndarray) for img in imgs]):
                # convert (h, w, c) to (c, h, w)
                imgs = [torch.from_numpy(img.transpose((2, 0, 1))).unsqueeze(0) for img in imgs]
            elif all([isinstance(img, torch.Tensor) for img in imgs]):
                imgs = [img.unsqueeze(0) for img in imgs]
            else:
                raise ValueError('imgs must be list of ndarray or Tensor')
            imgs = torch.cat(imgs, dim=0)
        else:
            raise ValueError('imgs must be list, ndarray or Tensor')

        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0) # shape = (1(=batch), seq_num*c(=c), h, w)

        if self.training:
            logging.info('Switched to eval mode')
            self.eval()

        heatmaps = self(imgs)
        ret = self.detector(heatmaps)

    def load_weights_from_origin(self, path):
        src_state_dict = torch.load(path)

        dest_state_dict = self.state_dict()
        for src_name, src_val in src_state_dict.items():
            if src_name in dest_state_dict.keys():
                dest_state_dict[src_name] = src_val

        self.load_state_dict(dest_state_dict)

        print('loaded from {}'.format(path))