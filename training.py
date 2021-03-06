from tracknet.tracknet import TrackNet
from tracknet.train import *
from tracknet.core.loss import CrossEntropy
from data.datasets import AllTrackNetTennis
from data.transforms import GaussianHeatMap, LossHeatMap

from torch.nn import CrossEntropyLoss
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

    model = TrackNet(image_shape=(360, 640, 3), seq_num=3, batch_norm=True)
    print(model)
    #import torch
    #torch.save(model.state_dict(), './weights/tnet_init.pth')

    loss_func = CrossEntropy()
    optimizer = Adadelta(model.parameters(), lr=1.0)
    trainer = Trainer(model, loss_func=loss_func, optimizer=optimizer, scheduler=None)

    trainer.train(1000, train_loader, savemodelname='tracknet', checkpoints_epoch_interval=50)