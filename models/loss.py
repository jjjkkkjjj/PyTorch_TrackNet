from torch import nn
import torch

class CrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        batch_num = input.shape[0]
        loss = -(target * torch.log(input)).view(batch_num, -1).mean(dim=1)
        return loss.sum()