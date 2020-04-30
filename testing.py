from tracknet.tracknet import TrackNet
from tracknet.train import *
from tracknet.core.loss import CrossEntropy
from data.datasets import AllTrackNetTennis
from data.transforms import GaussianHeatMap, LossHeatMap

from torch.utils.data import DataLoader
from torchvision.transforms import *
from torch.optim.adadelta import Adadelta


if __name__ == '__main__':
    transform = Compose([
        Resize((360, 640)),
        ToTensor()
    ])
    target_transform = Compose([
        GaussianHeatMap((360, 640), sigma2=10, threshold=128),
        LossHeatMap(256)
    ])

    train_dataset = AllTrackNetTennis(seq_num=3, transform=transform, target_transform=target_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=2,
                              shuffle=True)

    model = TrackNet(image_shape=(360, 640, 3), seq_num=3, batch_norm=False)
    print(model)

    model.load_weights_from_origin('./weights/model3-converted.pth')
    model.eval()

    data1, data2 = train_dataset[55], train_dataset[57]
    imgs = [data1[0], data2[0]]
    model.inference(imgs)