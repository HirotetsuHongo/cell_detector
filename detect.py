import os
import sys
import shutil as sh


def detect(image_path, output_path):
    pipe_path = 'workspace/pipe'

    if not os.path.exists(pipe_path):
        print('{} is not found.'.format(pipe_path))

    # path in workspace
    image_path2 = os.path.join('images', os.path.basename(image_path))
    output_path2 = os.path.join('results', os.path.basename(output_path))
    image_path3 = os.path.join('workspace', image_path2)
    output_path3 = os.path.join('workspace', output_path2)

    # move image into workspace
    sh.copyfile(image_path, image_path3)

    # detect
    with open(pipe_path, 'w') as pipe:
        pipe.write('detect {} {}'.format(image_path2, output_path2))
    # sleep
    while not os.path.exists(output_path3):
        continue
    print('Completed "detect {} {}".'.format(image_path, output_path))

    # postprocess
    os.remove(image_path3)
    sh.move(output_path3, output_path)


def main():
    # assert input
    if len(sys.argv) != 3:
        print('usage: {} image_path output_path'.format(sys.argv[0]))

    # initialize
    if not os.path.exists('workspace/images'):
        if os.path.isdir('workspace/images'):
            print('workspace/images is not a direcotry')
            return
        else:
            os.mkdir('workspace/images')
    if not os.path.exists('workspace/results'):
        if os.path.isdir('workspace/results'):
            print('workspace/results is not a direcotry')
            return
        else:
            os.mkdir('workspace/results')

    # get arguments
    image_path = sys.argv[1]
    output_path = sys.argv[2]

    detect(image_path, output_path)


if __name__ == '__main__':
    main()
