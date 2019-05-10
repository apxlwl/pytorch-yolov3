from options import Options
from yolo.models.yolov3 import Yolonet
from trainers import get_trainer
import json
import os
from torch import optim

opt = Options()
args = opt.opt
args.experiment_name = 'voc_608_darknet_foreground'
#args.gpu='0'
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
args.dataset_name='VOC'
args.dataset_root='/home/gwl/datasets/VOCdevkit'
# args.dataset_root='/disk3/datasets/voc/'
args.batch_size = 12
args.fliptest=False
args.multitest=False
args.net_size=480
args.resume = 'load_darknet'
#args.resume = 'load_yolov3'
# args.resume = 'best'
args.total_epoch=50
net = Yolonet(n_classes=1).cuda()
optimizer = optim.Adam(net.parameters(),lr=args.lr_initial,weight_decay=4e-05)
scheduler=optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,40], gamma=0.1)

_Trainer = get_trainer(dataset=args.dataset_name)(args=args,
                   model=net,
                   optimizer=optimizer,
                   lrscheduler=scheduler
                   )
if args.do_inference:
  _Trainer._inference_epoch(imgdir='input_imgdir',outdir='output_imgdir')
if args.do_test:
  _Trainer._valid_epoch(multiscale=args.multitest,flip=args.fliptest)
else:
  _Trainer.train()

  #
