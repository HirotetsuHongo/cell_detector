import torch


def bbox_iou(bboxes1, bboxes2):
    x1, y1 = bboxes1[:, 0], bboxes1[:, 1]
    w1, h1 = bboxes1[:, 2], bboxes1[:, 3]
    x2, y2 = bboxes2[:, 0], bboxes2[:, 1]
    w2, h2 = bboxes2[:, 2], bboxes2[:, 3]

    xmin1 = x1 - w1 * 0.5
    ymin1 = y1 - w1 * 0.5
    xmax1 = x1 + w1 * 0.5
    ymax1 = y1 + w1 * 0.5
    xmin2 = x2 - w2 * 0.5
    ymin2 = y2 - w2 * 0.5
    xmax2 = x2 + w2 * 0.5
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

    return iou


def loss(detected_bboxes, correct_bboxes):
    return
