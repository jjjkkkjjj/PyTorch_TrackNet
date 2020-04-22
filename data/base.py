from torch.utils.data import Dataset
from glob import glob
import os

from ._utils import _thisdir, _generate_annotaion_xml

class TrackNetTennisDataset(Dataset):

    def __init__(self, games_dir, transform=None):
        self.transform = transform
        self._games_dir = games_dir

        # create alllabels.csv
        _generate_annotaion_xml(update=False)
        # allpath -> game_dir -> indiceslist