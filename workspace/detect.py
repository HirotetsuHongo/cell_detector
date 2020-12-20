import model
import preprocess as pre
import postprocess as post
import config as cfg

import torch

import sys


def main():
    # assertion
    if len(sys.argv) != 3:
        print('usage: {} image_path output_path'.format(sys.argv[0]))
        return

    # constants
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    num_channels = cfg.config['num_channels']
    classes = cfg.config['classes']
    num_classes = len(classes)
    height = cfg.config['height']
    width = cfg.config['width']
    anchors = cfg.config['anchors']
    nms_iou = cfg.config['NMS_IoU']
    objectness = cfg.config['objectness']
    cuda = cfg.config['CUDA']
    weight_path = cfg.config['detect_weight']

    # network
    net = model.YOLOv3(num_channels, num_classes)
    if cuda:
        net = net.cuda()
    net.load_state_dict(torch.load(weight_path))
    print('Load weight from {}.'.format(weight_path))

    # detection
    image = pre.load_image(image_path, height, width, cuda)
    prediction = detect(net, image, anchors,
                        objectness, nms_iou,
                        cuda)

    # write prediction into output
    write_prediction(prediction, output_path, height, width)

    return


def detect(net, image, anchors, objectness, nms_iou, cuda):
    height = image.shape[1]
    width = image.shape[2]
    images = image.unsqueeze(0)
    predictions = net(images)
    predictions = post.postprocess(predictions,
                                   anchors,
                                   height, width,
                                   objectness, nms_iou,
                                   cuda)
    prediction = predictions[0]
    return prediction


def write_prediction(prediction, output_path, height, width):
    # create bboxes
    classes = post.select_classes(prediction)[1]
    xs = prediction[:, 0] / width
    ys = prediction[:, 1] / height
    ws = prediction[:, 2] / width
    hs = prediction[:, 3] / height
    objs = prediction[:, 4]
    bboxes = torch.cat((classes.unsqueeze(-1),
                        xs.unsqueeze(-1),
                        ys.unsqueeze(-1),
                        ws.unsqueeze(-1),
                        hs.unsqueeze(-1),
                        objs.unsqueeze(-1)),
                       -1)

    # constants
    num_bboxes = bboxes.shape[0]
    bbox_size = bboxes.shape[1]

    with open(output_path, 'w') as f:
        for i in range(num_bboxes):
            for j in range(bbox_size):
                if j == 0:
                    f.write('{:d}'.format(bboxes[i, j].int()))
                else:
                    f.write('{:.6f}'.format(bboxes[i, j]))
                if j != bbox_size - 1:
                    f.write(' ')
            if i != num_bboxes - 1:
                f.write('\n')

    return


if __name__ == '__main__':
    main()
