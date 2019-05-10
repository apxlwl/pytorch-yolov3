import torch
import torch.nn.functional as F


def process_output(feature_map, anchors, input_shape, num_classes, training=True):
  anchors = anchors.view(1, 1, 1, 3, 2)

  h, w = feature_map.shape[1:3]  # y,x
  bzsize = feature_map.shape[0]
  feature_map = feature_map.view([-1, h, w, 3, 5 + num_classes])
  box_centers, box_wh, conf_logits, prob_logits = torch.split(feature_map, [2, 2, 1, num_classes], dim=-1)  # xywh

  # get a meshgrid offset
  grid_x = torch.linspace(0, w - 1, w). \
    repeat(w, 1). \
    repeat(bzsize * 3, 1, 1). \
    view(bzsize, 3, w, w). \
    permute(0, 2, 3, 1).unsqueeze(-1).type(torch.cuda.FloatTensor)

  grid_y = torch.linspace(0, h - 1, h). \
    repeat(h, 1).t(). \
    repeat(bzsize * 3, 1, 1). \
    view(bzsize, 3, h, h). \
    permute(0, 2, 3, 1).unsqueeze(-1).type(torch.cuda.FloatTensor)
  xy_offset = torch.cat((grid_x, grid_y), -1)

  # normalize xy according to grid_size,equal to normalize to 416
  box_centers = (F.sigmoid(box_centers) + xy_offset) / torch.Tensor([w, h]).float().cuda()
  # normalize wh according to inputsize 416
  box_wh = torch.exp(box_wh) * anchors / input_shape
  box_conf = F.sigmoid(conf_logits)
  box_prob = F.sigmoid(prob_logits)

  if training:
    return xy_offset, feature_map, box_centers, box_wh, box_conf
  else:
    return box_centers, box_wh, box_conf, box_prob


def predict_yolo(feature_map_list, anchors, inputshape, imgshape, num_classes):
  boxes = []
  scores = []
  for idx in range(3):
    _feature, _anchor = feature_map_list[idx], anchors[idx]
    _h, _w = _feature.shape[-2:]
    _feature = _feature.view(3, num_classes + 5, _h, _w).permute(2, 3, 0, 1)
    _feature = _feature.unsqueeze(0)

    _boxes_center, _boxes_wh, _conf, _classes = process_output(_feature, _anchor, inputshape, training=False,
                                                               num_classes=num_classes)
    #_score = _conf * _classes
    #_score = _score.view([1, -1, num_classes])
    _score = _conf.view([1, -1, num_classes])
    
    _boxes_center = _boxes_center * imgshape
    _boxes_wh = _boxes_wh * imgshape

    _boxes = torch.cat((_boxes_center, _boxes_wh), dim=-1).view(1, -1, 4)
    boxes.append(_boxes)
    scores.append(_score)
  allboxes = torch.cat(boxes, dim=1)
  allscores = torch.cat(scores, dim=1)

  center_x, center_y, width, height = torch.split(allboxes, [1, 1, 1, 1], dim=-1)
  x_min = center_x - width / 2
  y_min = center_y - height / 2
  x_max = center_x + width / 2
  y_max = center_y + height / 2
  allboxes = torch.cat([x_min, y_min, x_max, y_max], dim=-1)
  return allboxes, allscores

def loss_yolo(feature_map_list, gt_list, anchors, inputshape, num_classes, use_focal=False):
  bzsize = feature_map_list[0].shape[0]
  bcelogit_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
  smooth_loss = torch.nn.SmoothL1Loss(reduction='none')
  batch_box = 0
  batch_conf = 0
  batch_class = 0
  for idx in range(3):
    _feature, _anchor, _gt = feature_map_list[idx], anchors[idx], gt_list[idx]

    _h, _w = _feature.shape[-2:]
    _object_mask = _gt[..., 4:5]
    _true_class_probs = _gt[..., 5:]
    _feature = _feature.view(bzsize, 3, num_classes + 5, _h, _w).permute(0, 3, 4, 1, 2)
    _xy_offset, _feature, _box_centers, _box_wh, _box_conf = process_output(_feature, _anchor, inputshape,num_classes=num_classes)
    # get ignoremask
    _valid_true_boxes = torch.masked_select(_gt[..., 0:4], _object_mask.byte()).reshape(-1, 4)
    if _valid_true_boxes.shape[0]>0:
      _valid_true_xy = _valid_true_boxes[:, 0:2]
      _valid_true_wh = _valid_true_boxes[:, 2:4]
      _ious = broadcast_iou(true_xy=_valid_true_xy, true_wh=_valid_true_wh,
                            pred_xy=_box_centers, pred_wh=_box_wh)

      _best_iou, _ = torch.max(_ious, dim=-1)
      _ignore_mask = _best_iou < 0.5
      _ignore_mask = _ignore_mask.unsqueeze(-1).float()
    else:
      _ignore_mask=torch.zeros_like(_object_mask).float()
    # manipulate the gt
    _raw_true_xy = _gt[..., :2] * torch.Tensor([_w, _h]).float().cuda() - _xy_offset
    _raw_true_wh = _gt[..., 2:4] / _anchor * inputshape
    _raw_true_wh = torch.where(_raw_true_wh == 0, torch.ones_like(_raw_true_wh), _raw_true_wh)
    _raw_true_wh = torch.log(_raw_true_wh)
    _box_loss_scale = 2 - _gt[..., 2:3] * _gt[..., 3:4]
    _xy_loss = 2.0 * 1.0 * _object_mask * _box_loss_scale * bcelogit_loss(input=_feature[..., 0:2], target=_raw_true_xy)
    _wh_loss = 2.0 * 1.5 * _object_mask * _box_loss_scale * smooth_loss(input=_feature[..., 2:4], target=_raw_true_wh)

    _conf_loss = (_object_mask * bcelogit_loss(target=_object_mask, input=_feature[..., 4:5]) +
                  (1 - _object_mask) * _ignore_mask * bcelogit_loss(target=_object_mask, input=_feature[..., 4:5]))

    _class_loss = _object_mask * bcelogit_loss(target=_true_class_probs,input=_feature[..., 5:])
    batch_box+=(_xy_loss+_wh_loss).sum()
    batch_conf+=_conf_loss.sum()
    batch_class+=_class_loss.sum()
  return batch_box,batch_conf,batch_class


def broadcast_iou(true_xy, true_wh, pred_xy, pred_wh):
  '''
  maintain an efficient way to calculate the ios matrix between ground truth true boxes and the predicted boxes
  note: here we only care about the size match
  '''
  # shape: [N, 13, 13, 3, 1, 2]
  pred_box_xy = pred_xy.unsqueeze(-2)
  pred_box_wh = pred_wh.unsqueeze(-2)

  # shape: [1, V, 2]
  true_box_xy = true_xy.unsqueeze(0)
  true_box_wh = true_wh.unsqueeze(0)
  # [N, 13, 13, 3, 1, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2]
  intersect_mins = torch.max(pred_box_xy - pred_box_wh / 2.,
                             true_box_xy - true_box_wh / 2.)
  intersect_maxs = torch.min(pred_box_xy + pred_box_wh / 2.,
                             true_box_xy + true_box_wh / 2.)
  intersect_wh = torch.max(intersect_maxs - intersect_mins, torch.zeros_like(intersect_mins))
  # shape: [N, 13, 13, 3, V]
  intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
  # shape: [N, 13, 13, 3, 1]
  pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
  # shape: [1, V]
  true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]

  # [N, 13, 13, 3, V]
  iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)
  return iou
