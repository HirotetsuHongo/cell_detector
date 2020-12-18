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
    write_prediction(prediction, output_path)

    return


def detect(net, image, anchors, objectness, nms_iou, cuda):
    height = image.shape[0]
    width = image.shape[1]
    images = image.unsqueeze(0)
    predictions = net(images)
    predictions = post.postprocess(predictions,
                                   anchors,
                                   height, width,
                                   objectness, nms_iou,
                                   cuda)
    prediction = predictions[0]
    return prediction


def write_prediction(prediction, output_path):
    prediction_size = prediction.shape[0]
    bbox_size = prediction.shape[1]
    with open(output_path, 'a') as f:
        for i in range(prediction_size):
            for j in range(bbox_size):
                f.write('{:.3f}'.format(prediction[i, j]))
                if j != bbox_size - 1:
                    f.write(' ')
            if i != prediction_size - 1:
                f.write('\n')
    return


if __name__ == '__main__':
    main()
