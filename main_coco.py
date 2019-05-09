from options import Options
from yolo.models.yolov3 import Yolonet
from trainers import get_trainer
import json
import os
from torch import optim
import torch
opt = Options()
args = opt.opt
args.experiment_name = 'coco_608'
# args.gpu='0'
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.dataset_root='/home/gwl/datasets/coco2017'
args.dataset_name='COCO'
args.batch_size = 12
args.net_size=608
args.fliptest=False
args.multitest=False
args.resume = 'load_darknet'
# args.resume = 'load_yolov3'
# args.resume = 'best'
net = Yolonet(n_classes=80).cuda()
net=net.cuda()
optimizer = optim.SGD(net.parameters(),lr=args.lr_initial,weight_decay=4e-05)
scheduler=optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70,90], gamma=0.1)

_Trainer = get_trainer(dataset=args.dataset_name)(args=args,
                   model=net,
                   optimizer=optimizer,
                   lrscheduler=scheduler
                   )
if args.do_inference:
  _Trainer._inference_epoch(imgdir='input_imgdir',outdir='output_imgdir')
elif args.do_test:
  _Trainer._valid_epoch(multiscale=args.multitest,flip=args.fliptest)
else:
  _Trainer.train()

  #
