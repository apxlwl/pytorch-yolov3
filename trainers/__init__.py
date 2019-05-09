from . import trainer_DUTS
from . import trainer_VOC
from . import trainer_COCO
def get_trainer(dataset):
  return eval("trainer_{}.Trainer".format(dataset))