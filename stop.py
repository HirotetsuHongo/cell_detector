import subprocess as sp
import os


def main():
    # delete a pipe
    pipe = 'workspace/pipe'
    if os.path.exists(pipe):
        os.remove(pipe)

    # stop container
    status_up = sp.Popen(['docker', 'ps'], stdout=sp.PIPE)
    status_up = sp.Popen(['grep', 'cell_detector'],
                         stdin=status_up.stdout)
    status_up = status_up is not None
    if status_up:
        sp.run('./commands/stop')


if __name__ == '__main__':
    main()
