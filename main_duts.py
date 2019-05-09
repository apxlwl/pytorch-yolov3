from options import Options
from yolo.models.yolov3 import Yolonet
from trainers import get_trainer
import json
import os
from torch import optim

opt = Options()
args = opt.opt
args.experiment_name = 'duts'
args.gpu='1'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.dataset_name='DUTS'
args.dataset_root='/home/gwl/datasets/saliency/DUTS'
args.lr_initial = 1e-4
args.total_epoch = 30
args.batch_size = 12
args.net_size= 320
args.fliptest=False
args.multitest=False
# args.resume = 'load_darknet'
args.resume = 'load_yolov3'
# args.resume = 'best'
# args.do_test = True
net = Yolonet(n_classes=1).cuda()
optimizer = optim.SGD(net.parameters(),lr=1e-4,weight_decay=4e-05)
scheduler=optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)

_Trainer = get_trainer(dataset=args.dataset_name)(args=args,
                   model=net,
                   optimizer=optimizer,
                   lrscheduler=scheduler
                   )
if args.do_test:
  _Trainer._valid_epoch(multiscale=args.multitest,flip=args.fliptest)
else:
  _Trainer.train()

  #

