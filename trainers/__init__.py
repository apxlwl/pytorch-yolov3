from . import trainer_DUTS
def get_trainer(dataset):
  return eval("trainer_{}.Trainer".format(dataset))