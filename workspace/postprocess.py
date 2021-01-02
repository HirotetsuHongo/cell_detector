import torch
import torch.nn.functional as F


def postprocess(predictions, anchors, height, width, confidency, iou, cuda):
    assert len(predictions) == len(anchors)
    predictions = [convert(prediction, ancs, height, width, cuda)
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
    with open('log.csv', 'w') as f:
        for pred in predictions[0]:
            for e in pred:
                f.write('{:.6f},'.format(e))
            f.write('\n')
    predictions = [suppress(prediction, confidency, iou)
                   for prediction in predictions]

    return predictions


def calculate_loss(predictions, targets, anchors,
                   height, width, cuda):
    predictions = [convert(prediction, ancs, height, width, cuda)
                   for (prediction, ancs) in zip(predictions, anchors)]

    loss = 0.0

    for prediction in predictions:
        assert prediction.shape[0] == len(targets)
        batch_size = prediction.shape[0]
        scale_height = prediction.shape[1]
        scale_width = prediction.shape[2]
        num_anchors = prediction.shape[3]
        for i in range(batch_size):
            loss += calc_loss(prediction[i].reshape(scale_height
                                                    * scale_width
                                                    * num_anchors,
                                                    -1),
                              targets[i],
                              height,
                              width,
                              scale_height,
                              scale_width,
                              num_anchors)
        loss /= batch_size

    return loss


def convert(prediction, anchors, height, width, cuda):
    batch_size = prediction.shape[0]
    h = prediction.shape[2]
    w = prediction.shape[3]
    stride_x = width / w
    stride_y = height / h
    num_anchors = len(anchors)
    cell_depth = prediction.shape[1] // num_anchors

    # reshape
    prediction = prediction.reshape(batch_size, num_anchors, cell_depth, h, w)
    prediction = prediction.permute(0, 1, 3, 4, 2)

    # sigmoid x and y
    prediction[:, :, :, :, 0:2] = torch.sigmoid(prediction[:, :, :, :, 0:2])

    # add offset
    grid_x = torch.arange(w)
    grid_y = torch.arange(h)
    if cuda:
        grid_x = grid_x.cuda()
        grid_y = grid_y.cuda()
    offset_y, offset_x = torch.meshgrid(grid_y, grid_x)
    offset_x = offset_x.unsqueeze(-1)
    offset_y = offset_y.unsqueeze(-1)
    offset = torch.cat((offset_x, offset_y), -1)
    prediction[:, :, :, :, 0:2] += offset

    # normalize x and y
    prediction[:, :, :, :, 0] *= stride_x
    prediction[:, :, :, :, 1] *= stride_y

    # permute
    prediction = prediction.permute(0, 2, 3, 1, 4)

    # log scale transform w and h
    prediction[:, :, :, :, 2] = torch.exp(prediction[:, :, :, :, 2])
    prediction[:, :, :, :, 3] = torch.exp(prediction[:, :, :, :, 3])

    # multiply anchors
    anchors = [[a[0] / stride_x, a[1] / stride_y] for a in anchors]
    anchors = torch.tensor(anchors)
    if cuda:
        anchors = anchors.cuda()
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


def suppress(prediction, confidency, iou):
    # get classes, class_scores and confidency
    class_scores, classes = select_classes(prediction)
    conf = class_scores * prediction[:, 4]

    # suppress by confidency threshold
    conf_mask = conf > confidency
    prediction = prediction[conf_mask]
    classes = classes[conf_mask]
    conf = conf[conf_mask]

    # non-maximum_suppress
    nms_cls_mask = classes.unsqueeze(1) == classes
    nms_conf_mask = conf.unsqueeze(1) < conf
    nms_iou_mask = bbox_iou(prediction.unsqueeze(1), prediction) > iou
    nms_mask = nms_cls_mask * nms_conf_mask * nms_iou_mask
    nms_mask = ~torch.any(nms_mask, 1)
    prediction = prediction[nms_mask]

    return prediction


def calc_loss(prediction, target, input_height, input_width,
              scale_height, scale_width, num_anchors):
    # constants
    lambda_coord = 5.0
    lambda_obj = 1.0
    lambda_noobj = 0.5
    lambda_cls = 1.0
    eps = 0.001

    # initial info
    num_prediction = prediction.shape[0]

    # get target position in feature map
    stride_x = input_width / scale_width
    stride_y = input_height / scale_height
    pos_x = target[:, 0] // stride_x
    pos_y = target[:, 1] // stride_y
    pos = (pos_x + pos_y * scale_width) * num_anchors
    pos = pos.long()
    pos = torch.cat([pos + i for i in range(num_anchors)], 0)

    # mask
    mask_noobj = torch.ones(num_prediction).bool()
    mask_noobj[pos] = False

    # prediction with or without object
    prediction_obj = prediction[pos]
    prediction_noobj = prediction[mask_noobj]

    # target repeat number of anchors
    target = target.repeat(3, 1)

    # loss
    loss_x = F.mse_loss(prediction_obj[:, 0] / input_width,
                        target[:, 0] / input_width,
                        reduction='sum')
    loss_y = F.mse_loss(prediction_obj[:, 1] / input_height,
                        target[:, 1] / input_height,
                        reduction='sum')
    loss_w = F.mse_loss(torch.sqrt(prediction_obj[:, 2] / input_width + eps),
                        torch.sqrt(target[:, 2] / input_width + eps),
                        reduction='sum')
    loss_h = F.mse_loss(torch.sqrt(prediction_obj[:, 3] / input_height + eps),
                        torch.sqrt(target[:, 3] / input_height + eps),
                        reduction='sum')
    loss_obj = F.mse_loss(prediction_obj[:, 4],
                          target[:, 4],
                          reduction='sum')
    loss_noobj = torch.sum(torch.square(prediction_noobj[:, 4:]))
    loss_cls = F.mse_loss(prediction_obj[:, 5:],
                          target[:, 5:],
                          reduction='sum')
    loss = \
        lambda_coord * (loss_x + loss_y + loss_w + loss_h) + \
        lambda_obj * loss_obj + \
        lambda_noobj * loss_noobj + \
        lambda_cls * loss_cls

    return loss


def select_classes(prediction):
    return torch.max(prediction[..., 5:], dim=-1)


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
    iou = areai / areau + eps

    return iou
