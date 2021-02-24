import subprocess as sp
import os


def main():
    # build container
    status_built = sp.run(['docker', 'image', 'ls'], stdout=sp.PIPE)
    status_built = status_built.stdout.decode('utf8').find('cell_detector')
    status_built = status_built >= 0
    if not status_built:
        sp.run('./commands/build')

    # run container
    status_exist = sp.run(['docker', 'ps', '-a'], stdout=sp.PIPE)
    status_exist = status_exist.stdout.decode('utf8').find('cell_detector')
    status_exist = status_exist >= 0
    if not status_exist:
        sp.run('./commands/run')
    else:
        # start container
        status_up = sp.run(['docker', 'ps'], stdout=sp.PIPE)
        status_up = status_up.stdout.decode('utf8').find('cell_detector')
        status_up = status_up >= 0
        if not status_up:
            sp.run('./commands/start')

    # make a pipe
    if not os.path.exists('workspace/pipe'):
        os.mkfifo('workspace/pipe')

    # make directories
    if not os.path.exists('workspace/dataset'):
        os.mkdir('workspace/dataset')
        os.mkdir('workspace/dataset/image')
        os.mkdir('workspace/dataset/train')
        os.mkdir('workspace/dataset/test')
        os.mkdir('workspace/dataset/weight')
    if not os.path.exists('images'):
        os.mkdir('images')
    if not os.path.exists('results'):
        os.mkdir('results')


if __name__ == '__main__':
    main()
