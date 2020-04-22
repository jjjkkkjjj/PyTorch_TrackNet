from models.tracknet import TrackNet
from models.train import Trainer
from data.datasets import AllTrackNetTennis
from data.transforms import GaussianHeatMap


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
        ToTensor()
    ])

    train_dataset = AllTrackNetTennis(transform=transform, target_transform=target_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=2,
                              shuffle=True)

    model = TrackNet(image_shape=(360, 640, 3), seq_num=3, batch_norm=True)
    print(model)

    optimizer = Adadelta(model.parameters(), lr=1.0)
    #trainer = Trainer(model, loss_func=, optimizer=optimizer, scheduler=None, gpu=True)