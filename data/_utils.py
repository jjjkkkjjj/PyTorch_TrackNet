import os
from pathlib import Path
import csv, logging
import cv2
from lxml import etree as ET
from collections import OrderedDict
import numpy as np

#_thisdir = os.path.dirname(__file__)
DATA_ROOT = os.path.join(os.path.expanduser("~"), 'data')

_header = ['file name', 'visibility', 'x-coordinate', 'y-coordinate,status']

def _generate_annotaion_xml(update=False):
    """
    Generate annotation xml file because csv format is not fitted to multi objects
    :return: xml_paths: list of xml paths

    ### create new xml files with voc style such like
    <annotation>
      <folder>game1/Clip1</folder>
      <filename>0000.jpg</filename>
      <source>
        <database>TrackNet</database>
        <annotation>TrackNet</annotation>
      </source>
      <owner>
        <name>Yu-Chuan Huang</name>
      </owner>
      <size>
        <width>1280</width>
        <height>720</height>
        <depth>3</depth>
      </size>
      <visibility>1</visibility>
      <status>0</status>
      <ball>
        <x>599</x>
        <y>423</y>
      </ball>
    </annotation>
    """
    base_path = os.path.join(DATA_ROOT, 'tennis_tracknet')

    # get xml and csv path
    xml_posixpaths = sorted(Path(base_path).rglob('*.xml'), key=lambda posixpath: str(posixpath))  # list of PosixPath class
    xml_paths = [str(xml_posixpath) for xml_posixpath in xml_posixpaths]

    csv_posixpaths = sorted(Path(base_path).rglob('*.csv'), key=lambda posixpath: str(posixpath))  # list of PosixPath class
    jpg_posixpaths = list(Path(base_path).rglob('*.jpg'))

    # if update is False and xml files exist same number as jpg's one, return
    if len(xml_posixpaths) == len(jpg_posixpaths) and not update:
        return xml_paths

    # remove existing all xml files
    for xml_path in xml_paths:
        os.remove(xml_path)

    source = OrderedDict()
    source['database'] = 'TrackNet'
    source['annotation'] = 'TrackNet'

    owner = OrderedDict()
    owner['name'] = 'Yu-Chuan Huang'

    logging.info('Generating xml files...')

    for i, csv_posixpath in enumerate(csv_posixpaths):
        relpath = os.path.relpath(csv_posixpath, base_path) # e.g. 'game1/Clip1/Label.csv'
        relpath = Path(relpath).parent # posixpath, e.g. 'game1/Clip1

        # read *.csv
        with open(csv_posixpath, 'r') as f:
            reader = csv.reader(f)
            # remove header
            rows = list(reader)[1:] # shape = (row number, column number=4=('file name', 'visibility', 'x-coordinate', 'y-coordinate,status')) but list

            for row in rows:
                # 'file name', 'visibility', 'x-coordinate', 'y-coordinate,status'
                filename = row[0]

                trees_orderedDict = OrderedDict()
                trees_orderedDict['folder'] = relpath
                trees_orderedDict['filename'] = filename
                trees_orderedDict['source'] = source
                trees_orderedDict['owner'] = owner
                trees_orderedDict['size'] = _compose_imagesize(os.path.join(base_path, relpath, filename))
                trees_orderedDict['visibility'] = row[1]
                trees_orderedDict['status'] = row[4]
                trees_orderedDict['ball'] = _compose_ball(x=row[2], y=row[3])

                et = _generate_ET(trees_orderedDict)
                # write
                filename, _ = os.path.splitext(filename)
                filename += '.xml'
                et.write(os.path.join(DATA_ROOT, base_path, relpath, filename), encoding='utf-8', pretty_print=True)
        logging.info('{}% {}/{}'.format(int(100*(float(i)/len(csv_posixpaths))), i + 1, len(csv_posixpaths)))

    logging.info('Finished!!!')

    # get xml and csv path again
    xml_posixpaths = sorted(Path(base_path).rglob('*.xml'),
                            key=lambda posixpath: str(posixpath))  # list of PosixPath class
    xml_paths = [str(xml_posixpath) for xml_posixpath in xml_posixpaths]
    return xml_paths

def _generate_ET(contents, root='annotation'):
    """
    :param contents: OrderedDict, key=subelement name, value=value or dict of subelement
    :param root: (Optional) root name, default is annotation
    :return:
    """

    def __recurrsive_ET(parent, value):
        if not isinstance(value, OrderedDict):
            parent.text = str(value)
            return
        else:
            for key, value in value.items():
                sub = ET.SubElement(parent, str(key))
                __recurrsive_ET(sub, value)


    root = ET.Element(root)
    et = ET.ElementTree(element=root)

    for key, value in contents.items():
        sub = ET.SubElement(root, str(key))
        __recurrsive_ET(sub, value)

    return et

def _compose_imagesize(path):
    """
    :param path: image path, str
    :return: ret, ordereddict including width, height and depth as key
    """
    img = cv2.imread(path)
    height, width, depth = img.shape

    ret = OrderedDict()
    ret['width'] = width
    ret['height'] = height
    ret['depth'] = depth

    return ret

def _compose_ball(x, y):
    """
    :param x: int
    :param y: int
    :return: ret, ordereddict including x and y as key
    """

    ret = OrderedDict()
    ret['x'] = x
    ret['y'] = y

    return ret

def get_image(xml_posixpath):
    """
    :param xml_posixpath: posixpath, xml path
    :return: img, bgr image shape = (h, w, c)
    """
    parser = ET.XMLParser(remove_blank_text=True)
    with open(xml_posixpath) as f:
        tree = ET.parse(f, parser=parser)

        basedir = xml_posixpath.parent
        filename = tree.find('filename').text

        path = os.path.join(basedir, filename)
        img = cv2.imread(path)

        return img

def get_balls(xml_posixpath):
    """
    :param xml_posixpath: posixpath, xml path
    :return: balls, ndarray, shape = (ball number, 2=(x, y))
    """
    parser = ET.XMLParser(remove_blank_text=True)
    with open(xml_posixpath) as f:
        tree = ET.parse(f, parser=parser)

        if int(tree.find('visibility').text) == 0:# no ball
            return np.array([[np.nan, np.nan]])

        balls = []
        for ball_tree in tree.iter('ball'):
            #print(ball.find('x').text, ball.find('y').text)
            x, y = int(ball_tree.find('x').text), int(ball_tree.find('y').text)

            balls += [[x, y]]

        return np.array(balls)