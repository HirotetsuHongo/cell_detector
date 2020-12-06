import torch
import torch.nn.functional as F


def postprocess(scales, anchors, input_height, input_width,
                objectness, iou, cuda):
    preds = convert_scales(scales, anchors, input_height, input_width, cuda)
    preds = [select_classes(pred) for pred in preds]
    preds = [suppress_prediction(pred, objectness, iou) for pred in preds]
    return preds


def calculate_loss(prediction, target, iou):
    num_prediction = prediction.shape[0]
    num_target = target.shape[0]

    iou_mask = bbox_iou(prediction.unsqueeze(1), target) >= iou
    class_mask = prediction[:, 4].unsqueeze(1) == target[:, 4]
    mask = torch.logical_and(iou_mask, class_mask)

    prediction = prediction.repeat(num_target, 1)
    prediction = prediction.reshape(num_target, num_prediction, -1)
    prediction = prediction.transpose(0, 1)
    prediction = prediction[mask]

    target = target.repeat(num_prediction, 1)
    target = target.reshape(num_prediction, num_target, -1)
    target = target[mask]

    # xywh loss
    loss = F.mse_loss(prediction[:, 0:4], target[:, 0:4], reduction='sum')

    # class loss

    return loss, prediction, target


def convert_scales(scales, anchorss, input_height, input_width, cuda):
    scales = [convert_scale(scale, anchors, input_height, input_width, cuda)
              for (scale, anchors) in zip(scales, anchorss)]
    predictions = torch.cat(scales, 1)
    predictions = [predictions[i] for i in range(predictions.shape[0])]
    return predictions


def select_classes(prediction):
    classes = torch.argmax(prediction[..., 5:], dim=-1)
    classes = classes.unsqueeze(-1)
    taple = (prediction[..., :5], classes, prediction[..., 5:])
    prediction = torch.cat(taple, -1)
    return prediction


def suppress_prediction(prediction, objectness, iou):
    # suppress by objectness threshold
    obj_mask = prediction[..., 4] >= objectness
    prediction = prediction[obj_mask]

    # non-maximum suppress
    iou_mask = bbox_iou(prediction.unsqueeze(-2), prediction) >= iou
    conf_mask = prediction[..., 4].unsqueeze(-2) >= prediction[..., 4]
    id_mask = prediction[..., 5].unsqueeze(-2) == prediction[..., 5]
    mask = torch.logical_and(iou_mask, conf_mask)
    mask = torch.logical_and(mask, id_mask)
    mask = torch.any(mask, -2)
    prediction = prediction[mask]

    return prediction


def bbox_iou(bboxes1, bboxes2):
    x1, y1 = bboxes1[..., 0], bboxes1[..., 1]
    w1, h1 = bboxes1[..., 2], bboxes1[..., 3]
    x2, y2 = bboxes2[..., 0], bboxes2[..., 1]
    w2, h2 = bboxes2[..., 2], bboxes2[..., 3]

    xmin1 = x1 - w1 * 0.5
    xmax1 = x1 + w1 * 0.5
    ymin1 = y1 - w1 * 0.5
    ymax1 = y1 + w1 * 0.5
    xmin2 = x2 - w2 * 0.5
    xmax2 = x2 + w2 * 0.5
    ymin2 = y2 - w2 * 0.5
    ymax2 = y2 + w2 * 0.5

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
    print("area1: {}".format(area1))
    print("area2: {}".format(area2))
    print("areai: {}".format(areai))

    return iou


def convert_scale(scale, anchors, input_height, input_width, cuda):
    batch_size = scale.shape[0]
    height = scale.shape[2]
    width = scale.shape[3]
    stride_x = input_width / width
    stride_y = input_height / height
    num_anchors = len(anchors)

    # reshape
    scale = scale.permute(0, 2, 3, 1)
    scale = scale.reshape(batch_size, height * width * num_anchors, -1)

    # sigmoid x and y
    scale[:, :, 0:2] = torch.sigmoid(scale[:, :, 0:2])

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
    centroid = centroid.repeat(1, num_anchors, 1)
    centroid = centroid.reshape(-1, 2)
    scale[:, :, 0:2] += centroid

    # normalize x and y
    scale[:, :, 0] *= stride_x
    scale[:, :, 1] *= stride_y

    # log scale transform w and h
    scale[:, :, 2] = torch.exp(scale[:, :, 2])
    scale[:, :, 3] = torch.exp(scale[:, :, 3])

    # multiply anchors
    anchors = [[a[0] / stride_x, a[1] / stride_y] for a in anchors]
    anchors = torch.tensor(anchors)
    if cuda:
        anchors = anchors.cuda()
    anchors = anchors.repeat(width * height, 1)
    scale[:, :, 2:4] *= anchors

    # sigmoid an objectness and class scores
    scale[:, :, 4:] = torch.sigmoid(scale[:, :, 4:])

    return scale
