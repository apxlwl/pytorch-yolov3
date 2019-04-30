import os
import os.path as osp
import cv2
import numpy as np
from dataset.pycocotools.coco import COCO
from dataset.augment import transform
from config import COCO_ANCHOR_608, COCO_ANCHOR_416, TRAIN_INPUT_SIZES_COCO
import random
import torch.utils.data as data
import torch

class COCOdataset(data.Dataset):
  def __init__(self, dataset_root, transform, subset, batchsize, netsize, istrain):
    self.dataset_root = dataset_root
    self.image_dir = "{}/images/{}2017".format(dataset_root, subset)
    self.coco = COCO("{}/annotations/instances_{}2017.json".format(dataset_root, subset))
    self.anchors = np.array(eval("COCO_ANCHOR_{}".format(netsize)))
    self.istrain = istrain
    self.netsize = netsize
    self.batch_size = batchsize
    # get the mapping from original category ids to labels
    self.cat_ids = self.coco.getCatIds()
    self.cat2label = {
      cat_id: i
      for i, cat_id in enumerate(self.cat_ids)
    }
    self.img_ids, self.img_infos = self._filter_imgs()
    self._transform = transform
    self.multisizes = TRAIN_INPUT_SIZES_COCO

  def _filter_imgs(self, min_size=32):
    # Filter images without ground truths.
    all_img_ids = list(set([_['image_id'] for _ in self.coco.anns.values()]))
    # Filter images too small.
    img_ids = []
    img_infos = []
    for i in all_img_ids:
      info = self.coco.loadImgs(i)[0]
      ann_ids = self.coco.getAnnIds(imgIds=i)
      ann_info = self.coco.loadAnns(ann_ids)
      ann = self._parse_ann_info(ann_info)
      if min(info['width'], info['height']) >= min_size and ann['labels'].shape[0] != 0:
        img_ids.append(i)
        img_infos.append(info)
    return img_ids, img_infos

  def _load_ann_info(self, idx):
    img_id = self.img_ids[idx]
    ann_ids = self.coco.getAnnIds(imgIds=img_id)
    ann_info = self.coco.loadAnns(ann_ids)
    return ann_info

  def _parse_ann_info(self, ann_info):
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []

    for i, ann in enumerate(ann_info):
      if ann.get('ignore', False):
        continue
      x1, y1, w, h = ann['bbox']
      if ann['area'] <= 0 or w < 1 or h < 1:
        continue
      bbox = [x1, y1, x1 + w, y1 + h]
      if ann['iscrowd']:
        gt_bboxes_ignore.append(bbox)
      else:
        gt_bboxes.append(bbox)
        gt_labels.append(self.cat2label[ann['category_id']])

    if gt_bboxes:
      gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
      gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
      gt_bboxes = np.zeros((0, 4), dtype=np.float32)
      gt_labels = np.array([], dtype=np.int64)

    if gt_bboxes_ignore:
      gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
      gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    ann = dict(bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

    return ann

  def __len__(self):
    return len(self.img_infos) // self.batch_size

  def __getitem__(self, index):
    if self.istrain:
      trainsize = random.choice(self.multisizes)
    else:
      trainsize = self.netsize

    return self._load_batch(index, trainsize)

  def _load_batch(self, idx_batch, random_trainsize):
    img_batch = []
    imgpath_batch = []
    annpath_batch = []
    pad_scale_batch = []
    ori_shape_batch = []
    grid0_batch = []
    grid1_batch = []
    grid2_batch = []
    for idx in range(self.batch_size):
      img_info = self.img_infos[idx_batch * self.batch_size + idx]
      ann_info = self._load_ann_info(idx_batch * self.batch_size + idx)
      # load the image.
      img = cv2.imread(osp.join(self.image_dir, img_info['file_name']), cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      ori_shape = img.shape[:2][::-1] #yx-->xy

      # Load the annotation.
      ann = self._parse_ann_info(ann_info)
      bboxes = ann['bboxes']  # [x1,y1,x2,y2]
      labels = ann['labels']
      img, bboxes = self._transform(random_trainsize, random_trainsize, img, bboxes)
      list_grids = transform.preprocess(bboxes, labels, img.shape[:2], class_num=80, anchors=self.anchors)
      pad_scale = (1, 1)
      img_batch.append(img)
      imgpath_batch.append(osp.join(self.image_dir, img_info['file_name']))
      annpath_batch.append(osp.join(self.image_dir, img_info['file_name']))
      ori_shape_batch.append(ori_shape)
      pad_scale_batch.append(pad_scale)
      grid0_batch.append(list_grids[0])
      grid1_batch.append(list_grids[1])
      grid2_batch.append(list_grids[2])

    return torch.from_numpy(np.array(img_batch).transpose((0,3,1,2)).astype(np.float32)), \
           imgpath_batch, \
           annpath_batch, \
           torch.from_numpy(np.array(pad_scale_batch).astype(np.float32)), \
           torch.from_numpy(np.array(ori_shape_batch).astype(np.float32)), \
           torch.from_numpy(np.array(grid0_batch).astype(np.float32)), \
           torch.from_numpy(np.array(grid1_batch).astype(np.float32)), \
           torch.from_numpy(np.array(grid2_batch).astype(np.float32))


def get_dataset(dataset_root, batch_size, net_size):
  datatransform = transform.YOLO3DefaultValTransform(mean=(0, 0, 0), std=(1, 1, 1))
  valset = COCOdataset(dataset_root, datatransform, subset='val', batchsize=batch_size, netsize=net_size,istrain=False)
  valset = torch.utils.data.DataLoader(dataset=valset,batch_size=1,shuffle=False,num_workers=4,pin_memory=True)

  return valset,valset
  datatransform = transform.YOLO3DefaultTrainTransform(mean=(0, 0, 0), std=(1, 1, 1))
  trainset = COCOdataset(dataset_root, datatransform, subset='train', shuffle=True, batchsize=batch_size,
                         netsize=net_size)
  trainset = tf.data.Dataset.from_generator(trainset,
                                            ((tf.float32, tf.string, tf.string, tf.float32, tf.float32, tf.float32,
                                              tf.float32,
                                              tf.float32)))
  # be careful to drop the last smaller batch if using tf.function
  trainset = trainset.batch(1, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

  return trainset, valset


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  train, val = get_dataset('/home/gwl/datasets/coco2017', 8, 416)
  for input in val:
    imgs=input[0]
    print(input[1][0])
    imgs=imgs.squeeze(0).permute(0,2,3,1)
    for img in imgs:
      plt.imshow(img)
      plt.show()
    assert 0
