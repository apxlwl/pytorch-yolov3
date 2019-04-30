from options import Options
from yolo.models.yolov3 import Yolonet
from trainers.trainer_voc import Trainer
import json
import os
from torch import optim
opt = Options()
args = opt.opt
args.experiment_name = 'voc_480_darknet'
args.gpu='0'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
args.dataset_name='VOC'
args.dataset_root='/home/gwl/datasets/VOCdevkit'
args.lr_initial = 1e-4
args.total_epoch = 250
args.log_iter = 5000
args.batch_size = 48
args.net_size= 480
args.fliptest=True
args.multitest=True
# args.resume = 'load_darknet'
# args.resume = 'load_yolov3'
args.resume = 'best'
args.do_test = True
net = Yolonet(n_classes=20).cuda()
# for k,v in net.state_dict().items():
#   print(k)
# assert 0
optimizer = optim.SGD(net.parameters(),lr=1e-4,weight_decay=4e-05)
scheduler=optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)
_Trainer = Trainer(args=args,
                   model=net,
                   optimizer=optimizer,
                   lrscheduler=scheduler
                   )
if args.do_test:
  _Trainer._valid_epoch(multiscale=args.multitest,flip=args.fliptest)
else:
  _Trainer.train()

  #
