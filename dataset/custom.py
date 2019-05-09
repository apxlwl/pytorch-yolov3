import numpy as np
from utils.dataset_util import PascalVocXmlParser
import cv2
from dataset.augment import transform
import os
from config import *
import random
import torch
from torch.utils.data import DataLoader
from os.path import join as osp
class CustomDataset:
  def __init__(self, dataset_name,dataset_root, transform, subset, batchsize, netsize, istrain):
    self.labels = eval("{}_LABEL".format(dataset_name))
    self.anchors = np.array(eval("{}_ANCHOR_{}".format(dataset_name,netsize)))
    self._transform = transform
    self.anno_root=osp(dataset_root,subset,'annotations')
    self.img_root=osp(dataset_root,subset,'imgs')
    self._annotations= [osp(self.anno_root,filename) for filename in os.listdir(self.anno_root)]
    self.netsize = netsize
    self.batch_size = batchsize
    self.multisizes = TRAIN_INPUT_SIZES_VOC
    self.istrain = istrain

  def __len__(self):
    return len(self._annotations) // self.batch_size

  def _load_batch(self, idx_batch, random_trainsize):
    img_batch = []
    imgpath_batch = []
    annpath_batch = []
    ori_shape_batch = []
    grid0_batch = []
    grid1_batch = []
    grid2_batch = []
    for idx in range(self.batch_size):
      annpath = self._annotations[idx_batch * self.batch_size + idx]
      fname, bboxes, labels, _ = PascalVocXmlParser(annpath, self.labels).parse()
      imgpath=osp(self.img_root,fname)
      img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      ori_shape = img.shape[:2][::-1]  # yx-->xy
      # Load the annotation.
      img, bboxes = self._transform(random_trainsize, random_trainsize, img, bboxes)
      list_grids = transform.preprocess(bboxes, labels, img.shape[:2], class_num=len(self.labels), anchors=self.anchors)
      img_batch.append(img)
      imgpath_batch.append(imgpath)
      annpath_batch.append(annpath)
      ori_shape_batch.append(ori_shape)
      grid0_batch.append(list_grids[0])
      grid1_batch.append(list_grids[1])
      grid2_batch.append(list_grids[2])
    return torch.from_numpy(np.array(img_batch).transpose((0, 3, 1, 2)).astype(np.float32)), \
           imgpath_batch, \
           annpath_batch, \
           torch.from_numpy(np.array(ori_shape_batch).astype(np.float32)), \
           torch.from_numpy(np.array(grid0_batch).astype(np.float32)), \
           torch.from_numpy(np.array(grid1_batch).astype(np.float32)), \
           torch.from_numpy(np.array(grid2_batch).astype(np.float32))

  def __getitem__(self, item):
    if self.istrain:
      trainsize = random.choice(self.multisizes)
    else:
      trainsize = self.netsize

    return self._load_batch(item, trainsize)


def get_dataset(dataset_name,dataset_root, batch_size, net_size):
  subset='val'
  datatransform = transform.YOLO3DefaultValTransform(mean=(0, 0, 0), std=(1, 1, 1))
  valset = CustomDataset(dataset_name,dataset_root, datatransform, subset, batch_size, net_size, istrain=False)

  valset = DataLoader(dataset=valset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  subset = 'train'
  datatransform = transform.YOLO3DefaultTrainTransform(mean=(0, 0, 0), std=(1, 1, 1))
  trainset = CustomDataset(dataset_name,dataset_root, datatransform, subset, batch_size, net_size, istrain=True)
  trainset = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
  return trainset, valset

if __name__ == '__main__':
  train,val=get_dataset('DUTS','/home/gwl/datasets/saliency/DUTS',2,320)
  for i in val:
    print(i)
    assert 0

