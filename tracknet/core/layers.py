from torch import nn
import torch
import numpy as np
import cv2
from torch.nn import functional as F

class SoftMax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_num, _, h, w = x.shape
        x = x.view((batch_num, -1, h*w))

        ret = []
        for b in range(batch_num):
            ret += [F.softmax(x[b], dim=0).unsqueeze(0)]
        ret = torch.cat(ret, dim=0).view((batch_num, -1, h, w))
        return ret

class HeatMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: Tensor, passed by softmax. shape = (b, c, h, w)
        :return: heatmap, Tensor, shape = (b, h, w)
        """
        heatmaps = torch.argmax(x, dim=1)

        return heatmaps


class Detector(nn.Module):
    def __init__(self, threshold=128):
        super().__init__()

        self.threshold = threshold

    def forward(self, heatmaps):
        """
        :param heatmaps: Float Tensor, shape = (b, h, w)
        :return:
        """

        heatmaps = heatmaps.numpy()
        # convert 0 or 255 as first step
        heatmaps[heatmaps >= self.threshold] = 255
        heatmaps[heatmaps < self.threshold] = 0
        heatmaps = heatmaps.astype(np.uint8)

        batch_num = heatmaps.shape[0]

        for b in range(batch_num):
            heatmap = heatmaps[b]

            cv2.imshow("heatmap", heatmap)
            cv2.waitKey()
