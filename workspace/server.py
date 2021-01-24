import model
import preprocess as pre
import config as cfg

import detect
import train
import test

import torch


def main():
    # constants
    num_channels = cfg.config['num_channels']
    classes = cfg.config['classes']
    num_classes = len(classes)
    height = cfg.config['height']
    width = cfg.config['width']
    anchors = cfg.config['anchors']
    num_anchors = len(anchors[0])
    nms_iou = cfg.config['NMS_IoU']
    confidency = cfg.config['confidency']
    cuda = cfg.config['CUDA']
    weight_path = cfg.config['path']['detect_weight']

    # unloaded net
    net = None

    # main loop
    while True:
        # print usage
        print('Available commands:')
        print('  detect image_path output_path')
        print('  train')
        print('  test')
        print('  quit | q | exit')

        # get user input
        with open('pipe', 'r') as pipe:
            command = pipe.read().replace('\n', '')

        # condition to quit
        if command == 'q':
            break
        elif command == 'quit':
            break
        elif command == 'exit':
            break

        command = command.split(' ')

        # detect
        if command[0] == 'detect':
            if len(command) != 3:
                print('usage: detect image_path output_path')
            else:
                image_path = command[1]
                output_path = command[2]

                # load net if it is not loaded
                if net is None:
                    net = model.YOLOv3(num_channels, num_classes, num_anchors)
                    if cuda:
                        net = net.cuda()
                    net.load_state_dict(torch.load(weight_path))
                    print('Load weight from {}'.format(weight_path))

                # load image
                image = pre.load_image(image_path, height, width, cuda)

                # predict and write bbox
                prediction = detect.detect(net, image, anchors,
                                           confidency, nms_iou,
                                           cuda)
                detect.write_prediction(prediction, output_path, height, width)

        # train
        elif command[0] == 'train':
            if len(command) != 1:
                print('usage: train')
            train.main()

        # test
        elif command[0] == 'test':
            if len(command) != 1:
                print('usage: test')
            test.main()

        # show usage
        else:
            print('{}: Unknown command.'.format(command))


if __name__ == '__main__':
    main()
