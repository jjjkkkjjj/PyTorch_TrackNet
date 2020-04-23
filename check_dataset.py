from data.datasets import AllTrackNetTennis
from data.transforms import GaussianHeatMap, LossHeatMap

from torchvision.transforms import *

if __name__ == '__main__':
    transform = Compose([
        Resize((360, 640)),
        ToTensor()
    ])
    target_transform = Compose([
        GaussianHeatMap((360, 640), sigma2=10, threshold=128),
        LossHeatMap(256)
    ])

    dataset = AllTrackNetTennis(seq_num=3, transform=transform, target_transform=target_transform)
    a = dataset[0]
    # no ball
    #a = dataset[74]
    exit()