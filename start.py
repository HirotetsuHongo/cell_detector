import subprocess as sp
import os


def main():
    # start container
    status_up = sp.Popen(['docker', 'ps'], stdout=sp.PIPE)
    status_up = sp.Popen(['grep', 'cell_detector'],
                         stdin=status_up.stdout)
    status_up.wait()
    status_up = status_up.stdout is not None
    if not status_up:
        sp.run('./commands/start')
    else:
        # run container
        status_exist = sp.Popen(['docker', 'ps', '-a'], stdout=sp.PIPE)
        status_exist = sp.Popen(['grep', 'cell_detector'],
                                stdin=status_exist.stdout)
        status_exist.wait()
        status_exist = status_exist.stdout is not None
        if not status_exist:
            sp.run('./commands/run')

    # make a pipe
    if not os.path.exists('workspace/pipe'):
        os.mkfifo('workspace/pipe')


if __name__ == '__main__':
    main()
