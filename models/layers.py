from torch import nn

class HeatMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: Tensor, passed by softmax. shape = (h, w, c)
        :return: heatmap, Tensor, shape = (h, w)
        """
        a=0