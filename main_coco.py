from options import Options
from yolo.models.yolov3 import Yolonet
from trainers.trainer_voc import Trainer
import json
import os
from torch import optim

opt = Options()
args = opt.opt
args.experiment_name = 'voc_480'
args.gpu='0'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
args.dataset_root='/home/gwl/datasets/coco2017'
args.dataset_name='COCO'
args.batch_size = 12
args.fliptest=False
args.multitest=False
args.resume = 'load_darknet'
# args.resume = 'load_yolov3'
# args.resume = 'best'
net = Yolonet(n_classes=20).cuda()
optimizer = optim.SGD(net.parameters(),lr=args.lr_initial,weight_decay=4e-05)
scheduler=optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70,90], gamma=0.1)

_Trainer = Trainer(args=args,
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
