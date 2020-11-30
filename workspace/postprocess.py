import torch
import loss


def postprocess(scales, anchors, input_height, input_width,
                objectness_threshold, iou_threshold, cuda):
    """
    postprocesses of YOLOv3.
    """
    def reshape(scale, batch_size, height, width, num_anchors, bbox_size):
        scale = scale.transpose(1, 3)
        scale = scale.reshape(batch_size,
                              height * width * num_anchors,
                              bbox_size)
        return scale

    def sigmoid_objectness(scale):
        scale[:, :, 4] = torch.sigmoid(scale[:, :, 4])
        return scale

    def adjust_coordinate(scale, height, width, stride, num_anchors, anchors):
        # sigmoid x and y
        scale[:, :, 0:2] = torch.sigmoid(scale[:, :, 0:2])

        # add center coordinate of bbox
        grid_x = torch.arange(width)
        grid_y = torch.arange(height)
        if cuda:
            grid_x = grid_x.cuda()
            grid_y = grid_y.cuda()
        center_x, center_y = torch.meshgrid(grid_x, grid_y)
        center_x = center_x.unsqueeze(-1)
        center_y = center_y.unsqueeze(-1)
        center = torch.cat((center_x, center_y), -1)
        center = center.repeat(1, num_anchors, 1).reshape(-1, 2)
        scale[:, :, 0:2] += center

        # normalize x and y
        scale[:, :, 0] *= stride[0]
        scale[:, :, 1] *= stride[1]

        # log scale transform w, h
        scale[:, :, 2:4] = torch.exp(scale[:, :, 2:4])
        # use anchors
        anchors = [[a[0] / stride[0], a[1] / stride[1]] for a in anchors]
        anchors = torch.tensor(anchors)
        if cuda:
            anchors = anchors.cuda()
        anchors = anchors.repeat(width * height, 1)
        scale[:, :, 2:4] *= anchors

        # normalize w and h
        scale[:, :, 2] *= stride[0]
        scale[:, :, 3] *= stride[1]

        return scale

    def low_objectness_suppress(image, objectness_threshold):
        mask = image[:, 4] >= objectness_threshold
        return image[mask, :]

    def select_class(image):
        class_id = torch.max(image[:, 5:], 1)[1]
        class_id = class_id.unsqueeze(1)
        image = torch.cat((image[:, :5], class_id), 1)
        return image

    def non_maximum_suppress(image, iou_threshold):
        iou_mask = loss.bbox_iou(image.unsqueeze(-1), image) > iou_threshold
        conf_mask = image[:, 4].unsqueeze(-1) >= image[:, 4]
        id_mask = image[:, 5].unsqueeze(-1) == image[:, 5]
        mask = torch.logical_and(iou_mask, conf_mask)
        mask = torch.logical_and(mask, id_mask)
        mask = torch.any(mask, 1)
        return image[mask, :]

    scales = list(scales)

    # scale-wise processes
    for i in range(len(scales)):
        batch_size = scales[i].size(0)
        height = scales[i].size(2)
        width = scales[i].size(3)
        num_anchors = len(anchors[i])
        bbox_size = scales[i].size(1) // num_anchors
        stride = (input_width / width, input_height / height)

        scales[i] = reshape(scales[i],
                            batch_size, height, width, num_anchors, bbox_size)
        scales[i] = sigmoid_objectness(scales[i])
        scales[i] = adjust_coordinate(scales[i],
                                      height, width, stride,
                                      num_anchors, anchors[i])

    images = torch.cat(scales, 1)
    images = [images[i] for i in range(images.size(0))]

    # image-wise processes
    for i in range(len(images)):
        images[i] = low_objectness_suppress(images[i], objectness_threshold)
        images[i] = select_class(images[i])
        images[i] = non_maximum_suppress(images[i], iou_threshold)

    return images
