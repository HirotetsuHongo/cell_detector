import subprocess as sp
import os


def main():
    # run container
    status_exist = sp.Popen(['docker', 'ps', '-a'], stdout=sp.PIPE)
    status_exist = sp.Popen(['grep', 'cell_detector'],
                            stdin=status_exist.stdout)
    status_exist.wait()
    status_exist = status_exist is not None
    if not status_exist.stdout:
        sp.run('./commands/run')

    # start container
    status_up = sp.Popen(['docker', 'ps'], stdout=sp.PIPE)
    status_up = sp.Popen(['grep', 'cell_detector'],
                         stdin=status_up.stdout)
    status_up.wait()
    status_up = status_up is not None
    if not status_up:
        sp.run('./commands/start')

    # make a pipe
    os.mkfifo('workspace/pipe')


if __name__ == '__main__':
    main()
