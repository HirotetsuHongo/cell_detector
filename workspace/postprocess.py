import torch
import torch.nn.functional as F


def postprocess(predictions, anchors, height, width, confidency, iou, cuda):
    assert len(predictions) == len(anchors)
    predictions = [convert_coord(prediction, ancs, height, width, cuda)
                   for (prediction, ancs) in zip(predictions, anchors)]
    predictions = [prediction
                   .reshape(prediction.shape[0],
                            prediction.shape[1]
                            * prediction.shape[2]
                            * prediction.shape[3],
                            -1)
                   for prediction in predictions]
    predictions = concat_scale(predictions)
    predictions = separate_batch(predictions)
#    with open('log.csv', 'w') as f:
#        for pred in predictions:
#            for bbox in pred:
#                f.write(','.join(['{:.6f}'.format(x) for x in list(bbox)]))
#                f.write('\n')
    predictions = [suppress(prediction, confidency, iou, cuda)
                   for prediction in predictions]
    predictions = [convert_conf(prediction)
                   for prediction in predictions]

    return predictions


def calculate_loss(predictions, targets, anchors,
                   height, width, cuda):
    predictions = [convert(prediction, ancs, height, width, cuda)
                   for (prediction, ancs) in zip(predictions, anchors)]

    loss = 0.0

    for prediction, ancs in zip(predictions, anchors):
        assert prediction.shape[0] == len(targets)
        batch_size = prediction.shape[0]
        scale_height = prediction.shape[1]
        scale_width = prediction.shape[2]
        num_anchors = prediction.shape[3]
        assert len(ancs) == num_anchors
        ls = 0.0
        for i in range(batch_size):
            ls += calc_loss(prediction[i].reshape(scale_height
                                                  * scale_width
                                                  * num_anchors,
                                                  -1),
                            targets[i],
                            height,
                            width,
                            scale_height,
                            scale_width,
                            ancs,
                            cuda)
        loss += ls / batch_size

    loss /= len(predictions)

    return loss


def calculate_AP(predictions, targets, tp_iou, cuda):
    num_classes = predictions[0].shape[-1] - 5
    classes, class_scores = zip(*[select_classes(pred)
                                  if pred.shape[0] != 0
                                  else (pred[:, 0], pred[:, 0])
                                  for pred in predictions])
    confidencies = [pred[:, 4] * score
                    for pred, score in zip(predictions, class_scores)]
    classes_t = [torch.argmax(tagt[:, 5:], -1)
                 if tagt.shape[0] != 0 else tagt[:, 0]
                 for tagt in targets]

    # initialize AP
    AP = []

    for c in range(num_classes):
        # select positive prediction for class c from prediction
        positives = [pred[cls == c] for pred, cls in zip(predictions, classes)]
        trues = [tagt[cls_t == c] for tagt, cls_t in zip(targets, classes_t)]
        confs = [conf[cls == c] for conf, cls in zip(confidencies, classes)]
        true_positive_tables = [bbox_iou(pstv.unsqueeze(1), true) >= tp_iou
                                for pstv, true
                                in zip(positives, trues)]
        positive_truths = [torch.any(table, -1)
                           for table in true_positive_tables]

        # count TN
        num_true_negatives = [int(torch.count_nonzero(~torch.any(table, 0)))
                              for table in true_positive_tables]
        num_true_negative = sum(num_true_negatives)

        # concatnate confs, positives and positive_truths
        positive = torch.cat(positives, dim=0)
        positive_truth = torch.cat(positive_truths, dim=0)
        conf = torch.cat(confs, dim=0)

        # sort by class_score
        order = torch.argsort(conf)
        positive = positive[order]
        positive_truth = positive_truth[order]

        # calculate precisions
        indices = torch.arange(positive_truth.shape[0])
        precisions = [float(torch.count_nonzero(positive_truth[:i+1]) / (i+1))
                      for i in indices
                      if positive_truth[i]]
        precisions = [max(precisions[i:]) for i in range(len(precisions))]

        # calculate AP
        if precisions == []:
            AP.append(0)
        else:
            AP.append(sum(precisions) / (len(precisions) + num_true_negative))

    return AP


def convert(prediction, anchors, height, width, cuda):
    prediction = convert_coord(prediction, anchors, height, width, cuda)
    prediction = convert_conf(prediction)
    return prediction


def convert_coord(prediction, anchors, height, width, cuda):
    batch_size = prediction.shape[0]
    scale_height = prediction.shape[2]
    scale_width = prediction.shape[3]
    stride_x = width / scale_width
    stride_y = height / scale_height
    num_anchors = len(anchors)
    cell_depth = prediction.shape[1] // num_anchors

    # reshape
    prediction = prediction.permute(0, 2, 3, 1)
    prediction = prediction.reshape(batch_size,
                                    scale_height*scale_width*num_anchors,
                                    cell_depth)

    # sigmoid x and y
    prediction[..., 0:2] = torch.sigmoid(prediction[..., 0:2])

    # add offset
    offset_x = torch.arange(scale_width)
    offset_y = torch.arange(scale_height)
    if cuda:
        offset_x = offset_x.cuda()
        offset_y = offset_y.cuda()
    offset_x = offset_x.unsqueeze(0).repeat(scale_height, 1)
    offset_y = offset_y.unsqueeze(1).repeat(1, scale_width)
    offset = torch.cat((offset_x.unsqueeze(-1), offset_y.unsqueeze(-1)), -1)
    offset = offset.repeat(1, 1, num_anchors).reshape(-1, 2)
    prediction[..., 0:2] += offset

    # normalize x and y
    prediction[..., 0] *= stride_x
    prediction[..., 1] *= stride_y

    # log scale transform w and h
    prediction[..., 2] = torch.exp(prediction[..., 2])
    prediction[..., 3] = torch.exp(prediction[..., 3])

    # multiply anchors
    anchors = torch.tensor(anchors)
    if cuda:
        anchors = anchors.cuda()
    anchors = anchors.repeat(scale_width*scale_height, 1)
    prediction[..., 2:4] *= anchors

    # reshape
    prediction = prediction.reshape(batch_size,
                                    scale_height,
                                    scale_width,
                                    num_anchors,
                                    cell_depth)

    return prediction


def convert_conf(prediction):
    prediction[..., 4:] = torch.sigmoid(prediction[..., 4:])
    return prediction


def concat_scale(predictions):
    predictions = torch.cat(predictions, 1)
    return predictions


def separate_batch(predictions):
    batch_size = predictions.shape[0]
    predictions = [predictions[i] for i in range(batch_size)]
    return predictions


def suppress(prediction, confidency, iou, cuda):
    # get classes, class_scores and confidency
    classes, class_scores = select_classes(prediction)
    objectness = prediction[:, 4]

    # suppress by confidency threshold
    conf_mask = torch.sigmoid(objectness) * torch.sigmoid(class_scores) \
        > confidency
    prediction = prediction[conf_mask]
    classes = classes[conf_mask]
    class_scores = class_scores[conf_mask]
    objectness = objectness[conf_mask]
    conf = objectness * class_scores

    # non-maximum suppression
    nms_cls_mask = classes.unsqueeze(1) == classes
    nms_conf_mask = conf.unsqueeze(1) < conf
    nms_iou_mask = bbox_iou(prediction.unsqueeze(1), prediction) > iou
    nms_mask = nms_cls_mask * nms_conf_mask * nms_iou_mask
    nms_mask = ~torch.any(nms_mask, 1)
    prediction = prediction[nms_mask]

    return prediction


def calc_loss(prediction, target, input_height, input_width,
              scale_height, scale_width, anchors, cuda):
    # constants
    lambda_coord = 20.0
    lambda_obj = 1.0
    lambda_noobj = 0.1
    lambda_cls = 10.0
    eps = 0.1

    # initial info
    num_anchors = len(anchors)
    num_prediction = prediction.shape[0]

    # get stride
    stride_x = input_width / scale_width
    stride_y = input_height / scale_height

    # get target offset and index
    offset_x = target[:, 0] // stride_x
    offset_y = target[:, 1] // stride_y
    pos = (offset_x + offset_y * scale_width) * num_anchors
    pos = pos.long()
    pos = torch.cat([pos + i for i in range(num_anchors)], 0)

    # mask
    mask_noobj = torch.ones(num_prediction).bool()
    mask_noobj[pos] = False

    # prediction with or without object
    prediction_obj = prediction[pos]
    prediction_noobj = prediction[mask_noobj]

    # target repeat number of anchors
    target = target.repeat(num_anchors, 1)

    # loss
    loss_x = F.mse_loss(prediction_obj[:, 0],
                        target[:, 0],
                        reduction='sum') / (stride_x ** 2)
    loss_y = F.mse_loss(prediction_obj[:, 1],
                        target[:, 1],
                        reduction='sum') / (stride_y ** 2)
    loss_w = torch.log(F.mse_loss(prediction_obj[:, 2],
                                  target[:, 2],
                                  reduction='sum') + 1)
    loss_h = torch.log(F.mse_loss(prediction_obj[:, 3],
                                  target[:, 3],
                                  reduction='sum') + 1)
    loss_obj = F.mse_loss(torch.logit(prediction_obj[:, 4], eps),
                          torch.logit(target[:, 4], eps),
                          reduction='sum')
    loss_noobj = torch.sum(
                    torch.square(
                        torch.logit(
                            prediction_noobj[:, 4:], eps)))
    loss_cls = F.mse_loss(torch.logit(prediction_obj[:, 5:], eps),
                          torch.logit(target[:, 5:], eps),
                          reduction='sum')

    loss = \
        lambda_coord * (loss_x + loss_y + loss_w / 2 + loss_h / 2) + \
        lambda_obj * loss_obj + \
        lambda_noobj * loss_noobj + \
        lambda_cls * loss_cls

#    n = torch.abs(torch.randn(1))[0]
#    if n < 5.0e-4:
#        with open('loss_log_{}-{}-{}-{}.txt'
#                  .format(lambda_coord,
#                          lambda_obj,
#                          lambda_noobj,
#                          lambda_cls),
#                  'a') as f:
#            f.write('{} x {}\n'.format(stride_x, stride_y))
#            f.write('{}\n'.format(torch.cat((prediction_obj.unsqueeze(-1),
#                                             target.unsqueeze(-1)),
#                                            -1)))
#            f.write('{}\n'.format(torch.mean(prediction_noobj[:, 4])))
#            f.write('{}\n'.format(torch.sort(prediction_noobj[:, 4],
#                                             descending=True)[0][:5]))
#            f.write('{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'
#                    .format(loss_x, loss_y, loss_w, loss_h,
#                            loss_obj, loss_noobj, loss_cls))

    return loss


def focal_loss(prediction, target, input_height, input_width,
               scale_height, scale_width, anchors, cuda):
    # constants
    lambda_coord = 20.0
    lambda_obj = 1.0
    lambda_noobj = 0.1
    lambda_cls = 10.0
    gamma = 2.0
    eps = 0.001

    # initial info
    num_anchors = len(anchors)
    num_prediction = prediction.shape[0]

    # get stride
    stride_x = input_width / scale_width
    stride_y = input_height / scale_height

    # get target offset and index
    offset_x = target[:, 0] // stride_x
    offset_y = target[:, 1] // stride_y
    pos = (offset_x + offset_y * scale_width) * num_anchors
    pos = pos.long()
    pos = torch.cat([pos + i for i in range(num_anchors)], 0)

    # mask
    mask_noobj = torch.ones(num_prediction).bool()
    mask_noobj[pos] = False

    # prediction with or without object
    prediction_obj = prediction[pos]
    prediction_noobj = prediction[mask_noobj]

    # target repeat number of anchors
    target = target.repeat(num_anchors, 1)

    # loss
    loss_x = F.mse_loss(prediction_obj[:, 0],
                        target[:, 0],
                        reduction='sum') / (stride_x ** 2)
    loss_y = F.mse_loss(prediction_obj[:, 1],
                        target[:, 1],
                        reduction='sum') / (stride_y ** 2)
    loss_w = torch.log(F.mse_loss(prediction_obj[:, 2],
                                  target[:, 2],
                                  reduction='sum') + 1)
    loss_h = torch.log(F.mse_loss(prediction_obj[:, 3],
                                  target[:, 3],
                                  reduction='sum') + 1)
    loss_obj = ~ torch.sum(torch.pow(1.0 - prediction_obj[:, 4], gamma)
                           * torch.log(prediction_obj[:, 4] + eps))
    loss_noobj = ~ torch.sum(torch.pow(prediction_noobj[:, 4], gamma)
                             * torch.log(1.0 - prediction_noobj[:, 4] + eps))
    loss_cls = ~ torch.sum(target[:, 5:]
                           * torch.pow(1.0 - prediction_obj[:, 5:], gamma)
                           * torch.log(prediction_obj[:, 5:] + eps)
                           + (1.0 - target[:, 5:])
                           * torch.pow(prediction_obj[:, 5:], gamma)
                           * torch.log(1.0 - prediction_obj[:, 5:] + eps))

    loss = \
        lambda_coord * (loss_x + loss_y + loss_w / 2 + loss_h / 2) + \
        lambda_obj * loss_obj + \
        lambda_noobj * loss_noobj + \
        lambda_cls * loss_cls

    return loss


def select_classes(prediction):
    class_scores, classes = torch.max(prediction[..., 5:], dim=-1)
    return classes, class_scores


def bbox_iou(bboxes1, bboxes2):
    eps = 0.001

    x1, y1 = bboxes1[..., 0], bboxes1[..., 1]
    w1, h1 = bboxes1[..., 2], bboxes1[..., 3]
    x2, y2 = bboxes2[..., 0], bboxes2[..., 1]
    w2, h2 = bboxes2[..., 2], bboxes2[..., 3]

    xmin1 = x1 - (w1 * 0.5)
    xmax1 = x1 + (w1 * 0.5)
    ymin1 = y1 - (h1 * 0.5)
    ymax1 = y1 + (h1 * 0.5)
    xmin2 = x2 - (w2 * 0.5)
    xmax2 = x2 + (w2 * 0.5)
    ymin2 = y2 - (h2 * 0.5)
    ymax2 = y2 + (h2 * 0.5)

    xmini = torch.maximum(xmin1, xmin2)
    ymini = torch.maximum(ymin1, ymin2)
    xmaxi = torch.minimum(xmax1, xmax2)
    ymaxi = torch.minimum(ymax1, ymax2)

    wi = torch.clamp(xmaxi - xmini, min=0.)
    hi = torch.clamp(ymaxi - ymini, min=0.)

    area1 = w1 * h1
    area2 = w2 * h2
    areai = wi * hi
    areau = area1 + area2 - areai
    iou = areai / (areau + eps)

    return iou
