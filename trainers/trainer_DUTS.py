from trainers.base_trainer import BaseTrainer
from evaluator.customeval import EvaluatorDUTS

class Trainer(BaseTrainer):
  def __init__(self, args, model, optimizer,lrscheduler):
    super().__init__(args, model, optimizer,lrscheduler)

  def _get_loggers(self):
    super()._get_loggers()

    self.TESTevaluator = EvaluatorDUTS(anchors=self.anchors,
                                      cateNames=self.labels,
                                      rootpath=self.dataset_root,
                                      use_07_metric=False
                                      )
    self.logger_custom = ['mAP']+['AP@{}'.format(cls) for cls in self.labels]