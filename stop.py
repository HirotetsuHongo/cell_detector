import subprocess as sp
import os


def main():
    # start container
    status_up = sp.Popen(['docker', 'ps'], stdout=sp.PIPE)
    status_up = sp.Popen(['grep', 'cell_detector'],
                         stdin=status_up.stdout)
    status_up = status_up is not None
    if status_up:
        sp.run('./commands/stop')

    # make a pipe
    pipe = 'workspace/pipe'
    if os.path.exists(pipe):
        with open(pipe, 'w') as p:
            p.write('quit\n')
        os.remove(pipe)


if __name__ == '__main__':
    main()
