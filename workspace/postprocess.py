import torch
import torch.nn.functional as F
from functools import reduce


def postprocess(predictions, anchors, height, width, cuda):
    assert len(predictions) == len(anchors)
    predictions = [convert(prediction, ancs, height, width, cuda)
                   for (prediction, ancs) in zip(predictions, anchors)]
    predictions = concat_scale(predictions)
    predictions = separate_batch(predictions)
    predictions = [suppress(prediction) for prediction in predictions]
    predictions = [prediction[:, :5] for prediction in predictions]

    return predictions


def calculate_loss(predictions, targets, anchors, iou, height, width, cuda):
    assert len(predictions) == len(anchors)
    predictions = [convert(prediction, ancs, height, width, cuda)
                   for (prediction, ancs) in zip(predictions, anchors)]

    loss = torch.tensor(0)
    if cuda:
        loss = loss.cuda()

    for prediction in predictions:
        assert prediction.shape[0] == len(targets)
        batch_size = prediction.shape[0]
        for i in range(batch_size):
            loss += calc_loss(prediction[i], targets[i], iou)

    return loss


def convert(prediction, anchors, height, width, cuda):
    batch_size = prediction.shape[0]
    h = prediction.shape[2]
    w = prediction.shape[3]
    stride_x = width / w
    stride_y = height / h
    num_anchors = len(anchors)

    # reshape
    prediction = prediction.permute(0, 2, 3, 1)
    prediction = prediction.reshape(batch_size,
                                    height * width,
                                    num_anchors,
                                    -1)

    # sigmoid x and y
    prediction[:, :, :, 0:2] = torch.sigmoid(prediction[:, :, :, 0:2])

    # add centroid
    grid_x = torch.arange(width)
    grid_y = torch.arange(height)
    if cuda:
        grid_x = grid_x.cuda()
        grid_y = grid_y.cuda()
    centroid_x, centroid_y = torch.meshgrid(grid_x, grid_y)
    centroid_x = centroid_x.unsqueeze(-1)
    centroid_y = centroid_y.unsqueeze(-1)
    centroid = torch.cat((centroid_x, centroid_y), -1)
    centroid = centroid.reshape(-1, 2)
    prediction[:, :, :, 0:2] += centroid

    # normalize x and y
    prediction[:, :, :, 0] *= stride_x
    prediction[:, :, :, 1] *= stride_y

    # log scale transform w and h
    prediction[:, :, :, 2] = torch.exp(prediction[:, :, :, 2])
    prediction[:, :, :, 3] = torch.exp(prediction[:, :, :, 3])

    # multiply anchors
    anchors = [[a[0] / stride_x, a[1] / stride_y] for a in anchors]
    anchors = torch.tensor(anchors)
    if cuda:
        anchors = anchors.cuda()
    prediction[:, :, :, 2:4] *= anchors

    # sigmoid an objectness and class scores
    prediction[:, :, :, 4:] = torch.sigmoid(prediction[:, :, :, 4:])

    prediction = prediction.reshape(batch_size,
                                    height * width * num_anchors,
                                    -1)

    return prediction


def concat_scale(predictions):
    predictions = torch.cat(predictions, 1)
    return predictions


def separate_batch(predictions):
    batch_size = predictions.shape[0]
    predictions = [predictions[i] for i in range(batch_size)]
    return predictions


def suppress(prediction, objectness, iou):
    # suppress by objectness threshold
    objectness_mask = prediction[:, 4] > objectness
    prediction = prediction[objectness_mask]

    # non-maximum_suppress
    class_scores, classes = select_classes(prediction)
    class_scores = class_scores * prediction[:, 4]
    nms_cls_mask = classes.unsqueeze(1) == classes
    nms_obj_mask = class_scores.unsqueeze(1) < class_scores
    nms_iou_mask = bbox_iou(prediction.unsqueeze(1), prediction) > iou
    nms_mask = nms_cls_mask * nms_obj_mask * nms_iou_mask
    nms_mask = ~torch.any(nms_mask, 1)
    prediction = prediction[nms_mask]

    return prediction


def calc_loss(prediction, target, scale_height, scale_width):
    # constants
    lambda_coord = 5.0
    lambda_noobj = 0.5

    # initial info
    num_prediction = prediction.shape[0]
    num_target = target.shape[0]
    prediction_class_scores, prediction_classes = select_classes(prediction)
    target_class_scores, target_classes = select_classes(target)

    # IoU mask
    mask_obj = bbox_iou(prediction.unsqueeze(1), target)
    mask_noobj = ~torch.any(mask_obj, 1)

    # cells with object
    pred_obj = prediction.unsqueeze(1)
    pred_obj = pred_obj.repeat(1, num_target, 1)
    pred_obj = pred_obj[mask_obj]

    tgt_obj = target.unsqueeze(0)
    tgt_obj = tgt_obj.repeat(num_prediction, 1, 1)
    tgt_obj = tgt_obj[mask_obj]

    # cells without object
    pred_noobj = prediction[mask_noobj]

    # loss
    loss_xy = lambda_coord * F.mse_loss(pred_obj[:, 0:2], tgt_obj[:, 0:2])
    loss_wh = lambda_coord * F.mse_loss(torch.sqrt(pred_obj[:, 2:4]),
                                        torch.sqrt(pred_obj[:, 2:4]))
    loss_obj = F.mse_loss(pred_obj[:, 4], tgt_obj[:, 4])
    loss_noobj = lambda_noobj * torch.sum(torch.square(pred_noobj[:, 4]))
    loss_cls = F.mse_loss(pred_obj[:, 5:], tgt_obj[:, 5:])
    loss = loss_xy + loss_wh + loss_obj + loss_noobj + loss_cls

    return loss


def select_classes(prediction):
    return torch.max(prediction[..., 5:], dim=-1)


def bbox_iou(bboxes1, bboxes2):
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
    iou = areai / areau

    return iou
