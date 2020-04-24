from torch import nn
import torch

class CrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, predicts, target):
        # predicts: shape = (b,c,h,w)
        # scaling
        predicts = predicts / torch.sum(predicts, dim=1, keepdim=True)

        # avoid nan
        predicts = predicts.clamp(min=1e-15, max=1 - 1e-15)
        loss = -(target * torch.log(predicts))#.view(batch_num, -1).mean(dim=1)
        return loss.mean()