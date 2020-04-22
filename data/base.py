from torch.utils.data import Dataset
from pathlib import Path
import os, cv2
import numpy as np
from PIL import Image

from ._utils import _generate_annotaion_xml, _thisdir, get_image, get_balls

class TrackNetTennisDataset(Dataset):

    def __init__(self, games_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self._games_dir = games_dir

        # create *.xml and get path of *.xml
        _ = _generate_annotaion_xml(update=False)

        # list of posixpath
        self._anno_posixpaths = []
        for game_dir in self._games_dir:
            anno_posixpaths = sorted(Path(os.path.join(_thisdir, 'tennis_tracknet', game_dir)).rglob('*.xml'), key=lambda posixpath: str(posixpath))
            self._anno_posixpaths += anno_posixpaths

    #def _jpgpath(self, folder, filename):

    def __getitem__(self, index):
        """
        :param index: int
        :returns:
            img: array-like, shape = (h, w, c), rgb order
            target: array-like, shape = (ball number, 2=(x, y))
        """
        # get image
        xml_posixpath = self._anno_posixpaths[index]
        img = get_image(xml_posixpath)
        # convert to rgb and pillow image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # get balls coordinates
        # test_xml = _thisdir + '/test.xml'
        target = get_balls(xml_posixpath).astype(np.float32)
        # normalize target
        target[:, 0] /= img.width
        target[:, 1] /= img.height

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self._anno_posixpaths)

