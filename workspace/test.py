import model
import config as cfg
import preprocess as pre
import postprocess as post

import torch

import time


def main():
    # constants
    num_channels = cfg.config['num_channels']
    classes = cfg.config['classes']
    num_classes = len(classes)
    height = cfg.config['height']
    width = cfg.config['width']
    anchors = cfg.config['anchors']
    num_anchors = len(anchors[0])
    confidency = cfg.config['confidency']
    tp_iou = cfg.config['TP_IoU']
    nms_iou = cfg.config['NMS_IoU']
    cuda = cfg.config['CUDA']
    target_dir = cfg.config['path']['test']
    image_dir = cfg.config['path']['image']
    weight_dir = cfg.config['path']['weight_test']
    image_paths = pre.load_image_paths(image_dir, target_dir)
    target_paths = pre.load_dir_paths(target_dir)
    weight_paths = pre.load_dir_paths(weight_dir)
    num_images = len(image_paths)

    # network
    net = model.YOLOv3(num_channels, num_classes, num_anchors)
    if cuda:
        net = net.cuda()

    # calculate loss or mAP for each weights
    for weight_path in weight_paths:
        net.load_state_dict(torch.load(weight_path))
        net.eval()
        loss_giou = 0.0
        loss_obj = 0.0
        loss_cls = 0.0
        loss_blc = 0.0
        predictions = []
        targets = []
        t0 = time.time()

        for i in range(num_images):
            # load image and target
            image = pre.load_image(image_paths[i],
                                   height,
                                   width,
                                   cuda)
            image = image.unsqueeze(0)
            target = pre.load_targets(target_paths[i:i+1],
                                      num_classes,
                                      height,
                                      width,
                                      cuda)

            # predict bbox
            prediction = [pred.detach() for pred in net(image)]

            # calculate loss
            loss = post.calculate_loss(prediction,
                                       target,
                                       anchors,
                                       height,
                                       width,
                                       cuda).detach()
            loss_giou += float(loss[0])
            loss_obj += float(loss[1])
            loss_cls += float(loss[2])
            loss_blc += float(loss[3])

            # save prediction and target as numpy array
            prediction = post.postprocess(prediction,
                                          anchors,
                                          height,
                                          width,
                                          confidency,
                                          nms_iou,
                                          cuda)
            predictions.extend([pred.detach() for pred in prediction])
            targets.extend([tagt.detach() for tagt in target])

        # normalize loss
        loss_giou /= num_images
        loss_obj /= num_images
        loss_cls /= num_images
        loss_blc /= num_images

        # calculate AP
        AP = post.calculate_AP(predictions, targets, tp_iou, cuda)

        elapsed_time = time.time() - t0

        print(('Weight: {}, Elapsed Time: {:.2f}s, ' +
               # 'GIoU Loss: {:.2f}, ' +
               # 'Objectness Loss: {:.2f}, ' +
               # 'Class Loss: {:.2f}, ' +
               # 'Balance Loss: {:.2f}, ' +
               'Loss: {:.2f}, ' +
               'AP: ' + ', '.join(['{:.2f}'.format(ap * 100) for ap in AP])
               + ', '
               'mAP: {:.2f}'.format(100 * sum(AP) / len(AP)))
              .format(weight_path, elapsed_time,
                      loss_giou, loss_obj, loss_cls, loss_blc,
                      loss_giou + loss_obj + loss_cls + loss_blc))


if __name__ == '__main__':
    main()
