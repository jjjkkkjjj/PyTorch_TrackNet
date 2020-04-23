import logging
from torch import nn
from collections import OrderedDict

from ._layerutils import *
from .layers import HeatMap

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

            ('softmax', nn.Softmax(dim=1)) # dim = 1 means along with channel of (b, c, h, w)
        ]
        self.feature_layers = nn.ModuleDict(feature_layers)

        self.heatmap = HeatMap()

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

            heatmap = self.heatmap(x) # shape = (h, w)
            return heatmap