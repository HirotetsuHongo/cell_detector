import subprocess as sp
import os


def main():
    # stop container
    status_up = sp.run(['docker', 'ps'], stdout=sp.PIPE)
    status_up = status_up.stdout.decode('utf-8').find('cell_detector')
    status_up = status_up >= 0
    if status_up:
        sp.run('./commands/stop')


if __name__ == '__main__':
    main()
