from .base import TrackNetTennisDataset
from ._utils import _thisdir

import os

class AllTrackNetTennis(TrackNetTennisDataset):
    def __init__(self, transform=None):
        games_dir = sorted(os.listdir(os.path.join(_thisdir, 'tennis_tracknet')))
        super().__init__(games_dir, transform=transform)

class Game1TrackNetTennis(TrackNetTennisDataset):
    def __init__(self, transform=None):
        super().__init__(games_dir=['game1'], transform=transform)