import os
from pathlib import Path
import csv
import numpy as np

_thisdir = os.path.dirname(__file__)
_header = ['relpath', 'file name', 'visibility', 'x-coordinate', 'y-coordinate,status']

def _generate_annotaion_xml(update=False):
    """
    Generate annotation xml file because csv format is not fitted to multi objects
    :return: alllabels_path: created csv's path
    """
    base_alllabels_path = os.path.join(_thisdir, 'tennis_tracknet')
    alllabels_path = os.path.join(base_alllabels_path, 'alllabels.csv')

    if os.path.exists(alllabels_path) and not update:
        return alllabels_path

    # remove alllabels.csv first
    os.remove(alllabels_path)

    # get labels path
    csv_posixpaths = sorted(Path(base_alllabels_path).rglob('*.csv'))  # list of PosixPath class

    # make new alllabels.csv
    with open(alllabels_path, 'w', newline="") as f:
        # header
        writer = csv.writer(f)
        writer.writerow(_header)


    for csv_posixpath in csv_posixpaths:
        relpath = os.path.relpath(csv_posixpath, base_alllabels_path) # e.g. 'game1/Clip1/Label.csv'
        relpath = str(Path(relpath).parent)
        # read *.csv
        with open(csv_posixpath, 'r') as f:
            reader = csv.reader(f)
            # remove header
            rows = np.array(list(reader))[1:] # shape = (row number, column number)
            row_num = rows.shape[0]
            # append relpath
            relpaths = np.broadcast_to(relpath, shape=(row_num, 1))
            rows = np.concatenate((relpaths, rows), axis=1).tolist()

            with open(alllabels_path, 'a') as fall:
                writer = csv.writer(fall)
                writer.writerows(rows)

    return alllabels_path