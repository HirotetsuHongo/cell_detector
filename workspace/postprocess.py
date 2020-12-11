import torch
import torch.nn.functional as F


def postprocess(predictions, anchors, height, width, cuda):
    assert len(predictions) == len(anchors)
    predictions = [convert(prediction, ancs, height, width, cuda)
                   for (prediction, ancs) in zip(predictions, anchors)]
    predictions = [prediction.reshape(-1, 9) for prediction in predictions]
    predictions = concat_scale(predictions)
    predictions = separate_batch(predictions)
    predictions = [suppress(prediction) for prediction in predictions]
    predictions = [prediction[:, :5] for prediction in predictions]

    return predictions


def calculate_loss(predictions, targets, anchors,
                   height, width, cuda):
    predictions = [convert(prediction, ancs, height, width, cuda)
                   for (prediction, ancs) in zip(predictions, anchors)]

    loss = torch.tensor(0.0)
    if cuda:
        loss = loss.cuda()

    for prediction in predictions:
        assert prediction.shape[0] == len(targets)
        batch_size = prediction.shape[0]
        scale_height = prediction.shape[1]
        scale_width = prediction.shape[2]
        num_anchors = prediction.shape[3]
        for i in range(batch_size):
            loss += calc_loss(prediction[i].reshape(-1, 9),
                              targets[i],
                              height,
                              width,
                              scale_height,
                              scale_width,
                              num_anchors) / batch_size

    return loss


def convert(prediction, anchors, height, width, cuda):
    batch_size = prediction.shape[0]
    h = prediction.shape[2]
    w = prediction.shape[3]
    stride_x = width / w
    stride_y = height / h
    num_anchors = len(anchors)
    assert prediction.shape[1] == num_anchors * 9

    # reshape
    prediction = prediction.permute(0, 2, 3, 1)
    prediction = prediction.reshape(batch_size, h, w, num_anchors, 9)

    # sigmoid x and y
    prediction[:, :, :, :, 0:2] = torch.sigmoid(prediction[:, :, :, :, 0:2])

    # add centroid
    grid_x = torch.arange(w)
    grid_y = torch.arange(h)
    if cuda:
        grid_x = grid_x.cuda()
        grid_y = grid_y.cuda()
    centroid_x, centroid_y = torch.meshgrid(grid_x, grid_y)
    centroid_x = centroid_x.unsqueeze(-1)
    centroid_y = centroid_y.unsqueeze(-1)
    centroid = torch.cat((centroid_x, centroid_y), -1)
    centroid = centroid.repeat(batch_size, num_anchors, 1)
    centroid = centroid.reshape(batch_size, h, w, num_anchors, 2)
    prediction[:, :, :, :, 0:2] += centroid

    # normalize x and y
    prediction[:, :, :, :, 0] *= stride_x
    prediction[:, :, :, :, 1] *= stride_y

    # log scale transform w and h
    prediction[:, :, :, :, 2] = torch.exp(prediction[:, :, :, :, 2])
    prediction[:, :, :, :, 3] = torch.exp(prediction[:, :, :, :, 3])

    # multiply anchors
    anchors = [[a[0] / stride_x, a[1] / stride_y] for a in anchors]
    anchors = torch.tensor(anchors)
    if cuda:
        anchors = anchors.cuda()
    anchors = anchors.repeat(batch_size * w * h, 1)
    anchors = anchors.reshape(batch_size, w, h, num_anchors, 2)
    prediction[:, :, :, :, 2:4] *= anchors

    # sigmoid an objectness and class scores
    prediction[:, :, :, :, 4:] = torch.sigmoid(prediction[:, :, :, :, 4:])

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


def calc_loss(prediction, target, input_height, input_width,
              scale_height, scale_width, num_anchors):
    # constants
    lambda_coord = 5.0
    lambda_noobj = 0.5

    # initial info
    num_prediction = prediction.shape[0]

    # get target position in feature map
    stride_x = input_width / scale_width
    stride_y = input_height / scale_height
    pos_x = target[:, 0] // stride_x
    pos_y = target[:, 1] // stride_y
    pos = (pos_x + pos_y * scale_width) * num_anchors
    pos = pos.long()
    pos = [pos + i for i in range(num_anchors)]
    pos = torch.cat(pos, 0)

    # mask
    mask_noobj = torch.ones(num_prediction).bool()
    mask_noobj[pos] = False

    # prediction with or without object
    prediction_obj = prediction[pos]
    prediction_noobj = prediction[mask_noobj]

    # target repeat number of anchors
    target = target.unsqueeze(1).repeat(1, 3, 1).reshape(-1, 9)

    # loss
    loss_xy = lambda_coord * F.mse_loss(prediction_obj[:, 0:2], target[:, 0:2])
    loss_wh = lambda_coord * F.mse_loss(torch.sqrt(prediction_obj[:, 2:4]),
                                        torch.sqrt(prediction_obj[:, 2:4]))
    loss_obj = F.mse_loss(prediction_obj[:, 4], target[:, 4])
    loss_noobj = lambda_noobj * torch.sum(torch.square(prediction_noobj[:, 4]))
    loss_cls = F.mse_loss(prediction_obj[:, 5:], target[:, 5:])
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
