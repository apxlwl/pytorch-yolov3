# pytorch-yolo3 

## Introduction
Pytorch implementation of YOLOv3. Tensorflow2.0 version can be found [here](https://github.com/wlguan/tensorflow2.0-yolov3)

## Quick Start 
1. Download yolov3.weights and darknet53.conv.74 from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Download COCO dataset
3. Modify the dataset root and weights root in the config file
```
python main_coco.py --resume load_yolov3 --do_test --net_size 608 --batch_size 8
```

## Training
1. run the following command to start training
```
python main_voc.py/main_coco.py --resume load_darknet --net_size 480 --batch_size 12
```

## Visualization
The Tensorboard is origanized like [TF-ObjectDection-API](https://github.com/tensorflow/models/tree/master/research/object_detection)


## Performance on VOC2007 Test(mAP)
Initial backbone weights | train scales| baseline|data augmentation | +multi test|+flip|
| ------ | ------ | ------ | ------ | ------ | ------ |
darknet53| 480|0.532|0.738|0.753|0.769
darknet53| 448,480,512|-|0.727|0.737|0.754
 coco pretrained | 448,480,512|-|0.817|0.834|0.845

Note: all experiments trained for 100 epochs with learning rate dropped 10 times at the 70 and 90 epoch.
## Supported Attributes
- [x] Data agumentation  
- [x] Multi-scale Training 
- [x] Multi-scale Testing(including flip)
- [x] Focal loss  
- [ ] ....
## TODO
- [x] Update VOC performance
- [ ] Update COCO performance
- [ ] Support distribute training
- [ ] Support Custom dataset  

## Reference
[gluon-cv](https://github.com/dmlc/gluon-cv)

[tf-eager-yolo3](https://github.com/penny4860/tf-eager-yolo3)

[keras-yolo3](https://github.com/qqwweee/keras-yolo3)

[stronger-yolo](https://github.com/Stinky-Tofu/Stronger-yolo)

[yolo3-pytorch](https://github.com/zhanghanduo/yolo3_pytorch)