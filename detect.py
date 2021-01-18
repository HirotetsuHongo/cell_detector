import os
import sys


def detect(image_path, output_path):
    pipe_path = 'workspace/pipe'

    if not os.path.exists(pipe_path):
        print('{} is not found.'.format(pipe_path))

    with open(pipe_path, 'w') as pipe:
        pipe.write('detect {} {}'.format(image_path, output_path))


def main():
    # assert input
    if len(sys.argv) != 3:
        print('usage: {} image_path output_path'.format(sys.argv[0]))

    # get arguments
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    if not os.path.exists(image_path):
        print('{} is not found.'.format(image_path))
        return

    # detect
    detect(image_path, output_path)


if __name__ == '__main__':
    main()
