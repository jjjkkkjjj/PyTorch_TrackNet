from .base import TrackNetTennisDataset
from ._utils import Config

import os

class AllTrackNetTennis(TrackNetTennisDataset):
    def __init__(self, seq_num, transform=None, target_transform=None):
        games_dir = sorted(os.listdir(os.path.join(Config.DATA_ROOT, 'tennis_tracknet')))
        super().__init__(seq_num, games_dir, transform=transform, target_transform=target_transform)

class Game1TrackNetTennis(TrackNetTennisDataset):
    def __init__(self, seq_num, transform=None, target_transform=None):
        super().__init__(seq_num, games_dir=['game1'], transform=transform, target_transform=target_transform)


class Game2TrackNetTennis(TrackNetTennisDataset):
    def __init__(self, seq_num, transform=None, target_transform=None):
        super().__init__(seq_num, games_dir=['game2'], transform=transform, target_transform=target_transform)


class Game3TrackNetTennis(TrackNetTennisDataset):
    def __init__(self, seq_num, transform=None, target_transform=None):
        super().__init__(seq_num, games_dir=['game3'], transform=transform, target_transform=target_transform)


class Game4TrackNetTennis(TrackNetTennisDataset):
    def __init__(self, seq_num, transform=None, target_transform=None):
        super().__init__(seq_num, games_dir=['game4'], transform=transform, target_transform=target_transform)


class Game5TrackNetTennis(TrackNetTennisDataset):
    def __init__(self, seq_num, transform=None, target_transform=None):
        super().__init__(seq_num, games_dir=['game5'], transform=transform, target_transform=target_transform)
        
        
class Game6TrackNetTennis(TrackNetTennisDataset):
    def __init__(self, seq_num, transform=None, target_transform=None):
        super().__init__(seq_num, games_dir=['game6'], transform=transform, target_transform=target_transform)
        
        
class Game7TrackNetTennis(TrackNetTennisDataset):
    def __init__(self, seq_num, transform=None, target_transform=None):
        super().__init__(seq_num, games_dir=['game7'], transform=transform, target_transform=target_transform)
        
        
class Game8TrackNetTennis(TrackNetTennisDataset):
    def __init__(self, seq_num, transform=None, target_transform=None):
        super().__init__(seq_num, games_dir=['game8'], transform=transform, target_transform=target_transform)
        
        
class Game9TrackNetTennis(TrackNetTennisDataset):
    def __init__(self, seq_num, transform=None, target_transform=None):
        super().__init__(seq_num, games_dir=['game9'], transform=transform, target_transform=target_transform)
        
        
class Game10TrackNetTennis(TrackNetTennisDataset):
    def __init__(self, seq_num, transform=None, target_transform=None):
        super().__init__(seq_num, games_dir=['game10'], transform=transform, target_transform=target_transform)