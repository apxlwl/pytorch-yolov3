import os
import glob
import os
import numpy as np
from xml.etree.ElementTree import parse
from pascal_voc_writer import Writer
from os.path import join as osp

def get_filelists(path, prefix, suffix):
  return glob.glob(os.path.join(path, '{}.{}'.format(prefix, suffix)))


# -*- coding: utf-8 -*-

class PascalVocXmlParser(object):
  """Parse annotation for 1-annotation file """

  def __init__(self, annfile, labels):
    self.annfile = annfile
    self.root = self._root_tag(self.annfile)
    self.tree = self._tree(self.annfile)
    self.labels = labels
  def parse(self):
    fname = self.get_fname()
    labels, diffcults = self.get_labels()
    boxes = self.get_boxes()
    # assert os.path.exists(fname),"file {} does not exist".format(fname)
    return fname, np.array(boxes), labels, diffcults

  def get_fname(self):
    return os.path.join(self.root.find("filename").text)

  def get_width(self):
    for elem in self.tree.iter():
      if 'width' in elem.tag:
        return int(elem.text)

  def get_height(self):
    for elem in self.tree.iter():
      if 'height' in elem.tag:
        return int(elem.text)

  def get_labels(self):
    labels = []
    difficult = []
    obj_tags = self.root.findall("object")
    for t in obj_tags:
      # labels.append(self.labels.index(t.find("name").text))
      labels.append(0)
      difficult.append(t.find("difficult").text)
    return labels, difficult

  def get_boxes(self):
    bbs = []
    obj_tags = self.root.findall("object")
    for t in obj_tags:
      box_tag = t.find("bndbox")
      x1 = box_tag.find("xmin").text
      y1 = box_tag.find("ymin").text
      x2 = box_tag.find("xmax").text
      y2 = box_tag.find("ymax").text
      box = np.array([(float(x1)), (float(y1)), (float(x2)), (float(y2))])
      bbs.append(box)
    bbs = np.array(bbs)
    return bbs

  def _root_tag(self, fname):
    tree = parse(fname)
    root = tree.getroot()
    return root

  def _tree(self, fname):
    tree = parse(fname)
    return tree


def txt2pascal(imgdir, txtfile, outputdir):
  f=open(txtfile,'r')
  lines=f.readlines()
  # Writer(path, width, height)
  for line in lines:
    filename,hw,*boxes=line.strip().split(' ')
    height,width=int(hw.split(',')[0]),int(hw.split(',')[1])
    writer = Writer(osp(imgdir,filename), width, height)
    # ::addObject(name, xmin, ymin, xmax, ymax)
    for box in boxes:
      boxinfo=box.split(',')
      boxinfo=[boxinfo[0]]+[int(coord) for coord in boxinfo[1:]]
      writer.addObject(*boxinfo)
    writer.save(osp(outputdir,filename.split('.')[0]+'.xml'))


if __name__ == '__main__':
  txt2pascal(imgdir='/home/gwl/datasets/saliency/DUTS/val/imgs', txtfile='/home/gwl/datasets/saliency/DUTS/val.txt',
             outputdir='/home/gwl/datasets/saliency/DUTS/val/annotations')
