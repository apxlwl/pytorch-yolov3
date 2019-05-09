from utils.util import ensure_dir
from dataset import get_COCO, get_VOC,get_DUTS
import os
import time
from config import *
from dataset import makeImgPyramids
from yolo import predict_yolo, loss_yolo
from utils.nms_utils import torch_nms, cpu_nms
from utils.util import module2weight
import numpy as np
from dataset import bbox_flip
from collections import defaultdict
import shutil
from tensorboardX import SummaryWriter
from utils.util import AverageMeter
import torch
import matplotlib.pyplot as plt
from yolo import load_darknet_weights
from torch.utils.data import DataLoader
from PIL import Image
class BaseTrainer:
  """
  Base class for all trainers
  """

  def __init__(self, args, model, optimizer,lrscheduler):
    self.args = args
    self.model = model
    self.optimizer = optimizer
    self.lr_scheduler=lrscheduler
    self.experiment_name = args.experiment_name
    self.dataset_name = args.dataset_name
    self.dataset_root = args.dataset_root
    self.batch_size = args.batch_size

    self.train_dataloader = None
    self.test_dataloader = None
    self.log_iter = self.args.log_iter
    self.net_size = self.args.net_size
    self.anchors = eval('{}_ANCHOR_{}'.format(self.args.dataset_name.upper(), self.net_size))
    self.anchors = torch.from_numpy(np.array(self.anchors)).view(3, 3, 2).float().cuda()
    self.labels = eval('{}_LABEL'.format(self.args.dataset_name.upper()))
    self.num_classes = len(self.labels)

    # logger attributes
    self.global_iter = 0
    self.global_epoch = 0
    self.TESTevaluator = None
    self.LossBox = None
    self.LossConf = None
    self.LossClass = None
    self.logger_custom = None
    self.metric_evaluate = None
    self.best_mAP=0
    # initialize
    self._get_model()
    self._get_SummaryWriter()
    self._get_dataset()
    self._get_loggers()

  def _save_ckpt(self, metric,name=None):
    state = {
      'epoch': self.global_epoch,
      'iter': self.global_iter,
      'state_dict': self.model.state_dict(),
      'opti_dict': self.optimizer.state_dict(),
      'metric':metric
    }
    if name == None:
      torch.save(state, os.path.join(self.save_path, 'checkpoint-{}.pth'.format(self.global_iter)))
    else:
      torch.save(state, os.path.join(self.save_path, 'checkpoint-{}.pth'.format(name)))
    print("save checkpoints at iter{}".format(self.global_iter))

  def _load_ckpt(self):
    if self.args.resume == "load_darknet":
      load_darknet_weights(self.model.backbone,os.path.join(self.args.pretrained_model,'darknet53.conv.74'))
    elif self.args.resume == "load_yolov3":
      load_darknet_weights(self.model.backbone,os.path.join(self.args.pretrained_model,'yolov3.weights'))
    else:  # iter or best
      ckptfile = torch.load(os.path.join(self.save_path, 'checkpoint-{}.pth'.format(self.args.resume)))
      self.model.load_state_dict(ckptfile['state_dict'])
      self.optimizer.load_state_dict(ckptfile['opti_dict'])
      self.global_epoch = ckptfile['epoch']
      self.global_iter = ckptfile['iter']
      # self.best_mAP=ckptfile['metric']
    print("successfully load checkpoint {}".format(self.args.resume))

  def _get_model(self):
    self.save_path = './checkpoints/{}/'.format(self.args.experiment_name)
    ensure_dir(self.save_path)
    self._prepare_device()
    if self.args.resume:
      self._load_ckpt()
    self.model = torch.nn.parallel.DataParallel(self.model)
    self.model.cuda()
  def _prepare_device(self):
    # TODO: add distributed training
    pass

  def _get_SummaryWriter(self):
    if not self.args.debug and not self.args.do_test:
      ensure_dir(os.path.join('./summary/', self.experiment_name))

      self.writer = SummaryWriter(log_dir='./summary/{}/{}'.format(self.experiment_name, time.strftime(
        "%m%d-%H-%M-%S", time.localtime(time.time()))))

  def _get_dataset(self):
    self.train_dataloader, self.test_dataloader = eval('get_{}'.format(self.dataset_name))(
      dataset_name=self.args.dataset_name,
      dataset_root=self.dataset_root,
      batch_size=self.args.batch_size,
      net_size=self.net_size
    )

  def train(self):
    for epoch in range(self.global_epoch, self.args.total_epoch):
      #lr scheduler
      self.lr_scheduler.step(epoch)
      lr_current = self.optimizer.param_groups[0]['lr']
      self.writer.add_scalar("learning_rate", lr_current, epoch)

      self.global_epoch += 1
      self._train_epoch()
      if epoch>2:
        results, imgs = self._valid_epoch(multiscale=False, flip=False)
        for k, v in zip(self.logger_custom, results):
          self.writer.add_scalar(k, v, global_step=self.global_iter)
        for k, v in self.logger_losses.items():
          self.writer.add_scalar(k, v.get_avg(), global_step=self.global_iter)
        for i in range(len(imgs)):
          self.writer.add_image("detections_{}".format(i), imgs[i].transpose(2, 0, 1),
                                global_step=self.global_iter)
        self._reset_loggers()
        if results[0] > self.best_mAP:
          self.best_mAP = results[0]
          self._save_ckpt(name='best',metric=self.best_mAP)
        if epoch%10==0:
          self._save_ckpt(metric=results[0])
  def _get_loggers(self):
    self.LossBox = AverageMeter()
    self.LossConf = AverageMeter()
    self.LossClass = AverageMeter()
    self.logger_losses = {}
    self.logger_losses.update({"lossBox": self.LossBox})
    self.logger_losses.update({"lossConf": self.LossConf})
    self.logger_losses.update({"lossClass": self.LossClass})

  def _reset_loggers(self):
    self.TESTevaluator.reset()
    self.LossClass.reset()
    self.LossConf.reset()
    self.LossBox.reset()

  def train_step(self, imgs, labels):
    imgs = imgs.cuda()
    labels = [label.cuda() for label in labels]
    outputs = self.model(imgs)

    loss_box, loss_conf, loss_class = loss_yolo(outputs, labels, anchors=self.anchors,
                                                inputshape=self.net_size,
                                                num_classes=self.num_classes)
    totalloss = (loss_box + loss_conf + loss_class) / imgs.shape[0]

    self.optimizer.zero_grad()
    totalloss.backward()
    self.optimizer.step()
    self.LossBox.update(loss_box.sum() / imgs.shape[0])
    self.LossConf.update(loss_conf.sum() / imgs.shape[0])
    self.LossClass.update(loss_class.sum() / imgs.shape[0])

  def _train_epoch(self):
    self.model.train()
    for idx_batch, inputs in enumerate(self.train_dataloader):
      if idx_batch==3:
        break
      inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
      img, _, _, _, *labels = inputs
      self.global_iter += 1
      if self.global_iter % 200 == 0:
        print(self.global_iter)
        for k, v in self.logger_losses.items():
          print(k, ":", v.get_avg())
      self.train_step(img, labels)

  #TODO merge the duplicated codes
  def _inference_epoch(self,imgdir,outdir=None,multiscale=True,flip=True):
    from dataset import get_imgdir
    from utils.visualize import visualize_boxes
    self.model.eval()
    dataloader = get_imgdir(imgdir, batch_size=8, net_size=self.net_size)
    for i,(imgpath,imgs,ori_shapes) in enumerate(dataloader):
      ori_shapes=ori_shapes.cuda()
      if not multiscale:
        INPUT_SIZES = [self.net_size]
      else:
        INPUT_SIZES = [self.net_size - 32, self.net_size, self.net_size + 32]
      pyramids = makeImgPyramids(imgs.numpy().transpose(0, 2, 3, 1), scales=INPUT_SIZES, flip=flip)
      # produce outputFeatures for each scale
      img2multi = defaultdict(list)
      for idx, pyramid in enumerate(pyramids):
        pyramid = torch.from_numpy(pyramid.transpose(0, 3, 1, 2)).cuda()
        with torch.no_grad():
          grids = self.model(pyramid)
        for imgidx in range(imgs.shape[0]):
          img2multi[imgidx].append([grid[imgidx] for grid in grids])

      # append prediction for each image per scale/flip
      for imgidx, scalegrids in img2multi.items():
        allboxes = []
        allscores = []
        for _grids, _scale in zip(scalegrids[:len(INPUT_SIZES)], INPUT_SIZES):
          _boxes, _scores = predict_yolo(_grids, self.anchors, _scale, ori_shapes[imgidx],
                                         num_classes=self.num_classes)
          allboxes.append(_boxes)
          allscores.append(_scores)
        if flip:
          for _grids, _scale in zip(scalegrids[len(INPUT_SIZES):], INPUT_SIZES):
            _boxes, _scores = predict_yolo(_grids, self.anchors, _scale, ori_shapes[imgidx],
                                           num_classes=self.num_classes)
            _boxes = bbox_flip(_boxes.squeeze(0), flip_x=True, size=ori_shapes[imgidx])
            _boxes = _boxes[np.newaxis, :]
            allboxes.append(_boxes)
            allscores.append(_scores)
        nms_boxes, nms_scores, nms_labels = torch_nms(torch.cat(allboxes, dim=1),
                                                    torch.cat(allscores, dim=1),
                                                    num_classes=self.num_classes)
        if nms_boxes is not None:
          detected_img=visualize_boxes(np.array(Image.open(imgpath[imgidx]).convert('RGB')),
                                       boxes=nms_boxes.cpu().numpy(),
                                       labels=nms_labels.cpu().numpy(),
                                       probs=nms_scores.cpu().numpy(),
                                       class_labels=self.labels)
          if outdir is not None:
            plt.imsave(os.path.join(outdir,imgpath[imgidx].split('/')[-1]),detected_img)

  def _valid_epoch(self, multiscale, flip):
    s=time.time()
    self.model.eval()
    for idx_batch, inputs in enumerate(self.test_dataloader):
      if idx_batch == self.args.valid_batch and not self.args.do_test:  # to save time
        break
      if idx_batch==3:
        break
      inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
      (imgs, imgpath, annpath, ori_shapes, *_) = inputs

      ori_shapes = ori_shapes.float().cuda()
      if not multiscale:
        INPUT_SIZES = [self.net_size]
      else:
        INPUT_SIZES = [self.net_size - 32, self.net_size, self.net_size + 32]
      pyramids = makeImgPyramids(imgs.numpy().transpose(0, 2, 3, 1), scales=INPUT_SIZES, flip=flip)
      # produce outputFeatures for each scale
      img2multi = defaultdict(list)
      for idx, pyramid in enumerate(pyramids):
        pyramid = torch.from_numpy(pyramid.transpose(0, 3, 1, 2)).cuda()
        with torch.no_grad():
          grids = self.model(pyramid)
        for imgidx in range(imgs.shape[0]):
          img2multi[imgidx].append([grid[imgidx] for grid in grids])

      # append prediction for each image per scale/flip
      for imgidx, scalegrids in img2multi.items():
        allboxes = []
        allscores = []
        for _grids, _scale in zip(scalegrids[:len(INPUT_SIZES)], INPUT_SIZES):
          _boxes, _scores = predict_yolo(_grids, self.anchors, _scale, ori_shapes[imgidx],
                                         num_classes=self.num_classes)
          allboxes.append(_boxes)
          allscores.append(_scores)
        if flip:
          for _grids, _scale in zip(scalegrids[len(INPUT_SIZES):], INPUT_SIZES):
            _boxes, _scores = predict_yolo(_grids, self.anchors, _scale, ori_shapes[imgidx],
                                           num_classes=self.num_classes)
            _boxes = bbox_flip(_boxes.squeeze(0), flip_x=True, size=ori_shapes[imgidx])
            _boxes = _boxes[np.newaxis, :]
            allboxes.append(_boxes)
            allscores.append(_scores)
        nms_boxes, nms_scores, nms_labels = torch_nms(torch.cat(allboxes, dim=1),
                                                    torch.cat(allscores, dim=1),
                                                    num_classes=self.num_classes)
        if nms_boxes is not None:
          boxes_np,scores_np,labels_np=nms_boxes.cpu().numpy(),nms_scores.cpu().numpy(),nms_labels.cpu().numpy()
          self.TESTevaluator.append(imgpath[imgidx][0],
                                    annpath[imgidx][0],
                                    boxes_np,
                                    scores_np,
                                    labels_np)
    results = self.TESTevaluator.evaluate()
    imgs = self.TESTevaluator.visual_imgs
    for k, v in zip(self.logger_custom, results):
      print("{}:{}".format(k, v))
    print("validation cost {} s".format(time.time()-s))
    return results, imgs
    # t=time.time()-s
    # print(t)
    # print(len(self.test_dataloader)/t)