from torch import nn
import torch

class CrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):

        return -(target * torch.log(input)).mean()