import subprocess as sp


def main():
    sp.run(['docker', 'exec', '-it', 'cell_detector',
            'pipenv', 'run', 'python', 'test.py'])


if __name__ == '__main__':
    main()

