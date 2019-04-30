import torch
import os
from collections import OrderedDict


def ensure_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)


def module2weight(moduledict):
  newdict = OrderedDict()
  for k, v in moduledict.items():
    newdict.update({k.replace('module.', ''): v})
  return newdict

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    # self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, temp_sum, n=1):
    # self.val = val
    self.sum += temp_sum
    self.count += n
    # self.avg = float(self.sum)/ float(self.count)

  def get_avg(self):
    return float(self.sum) / float(self.count)