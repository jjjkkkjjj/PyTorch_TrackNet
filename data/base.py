from torch.utils.data import Dataset
from pathlib import Path
import os, cv2
import numpy as np
from PIL import Image
import torch

from ._utils import _generate_annotaion_xml, _thisdir, get_image, get_balls

class TrackNetTennisDataset(Dataset):

    def __init__(self, seq_num, games_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self._seq_num = seq_num
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
            imgs: Tensor, shape = (h, w, c), c = rgb order * seq num
            targets: Tensor, shape = (ball number, 2=(x, y))
        """

        if len(self._anno_posixpaths) - self._seq_num < index:
            index = len(self._anno_posixpaths) - self._seq_num

        indices = []
        base_parent = self._anno_posixpaths[index].parent
        for i in range(self._seq_num):
            if base_parent == self._anno_posixpaths[index + i].parent:
                indices += [index]
            else:
                indices.insert(0, index - i)



        imgs, targets = [], []
        for i in indices:
            # get image
            xml_posixpath = self._anno_posixpaths[i]
            if base_parent != xml_posixpath.parent:
                raise ValueError('Image number in folder is insufficient to seq_num')

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

            imgs += [img]
            targets += [target]


        imgs = torch.cat(imgs, dim=0)
        targets = np.mean(np.concatenate(targets, axis=0), axis=0, keepdims=True)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return imgs, torch.from_numpy(targets)

    def __len__(self):
        return len(self._anno_posixpaths)

