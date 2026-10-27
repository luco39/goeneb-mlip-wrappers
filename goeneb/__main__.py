import argparse

from logging_module import setup_logger

logger = setup_logger()

from neb import main

def parse_args():
    """Parser function to read the user input."""
    parser = argparse.ArgumentParser(prog='GoeNEB',description=
                                     'Generating and optimizing a reaction pathway by NEB.')
    parser.add_argument('input_file', 
                        help='The name of the input file. The path to the working directory also works.')
    parser.add_argument('-v', '--verbose', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='How much output the program should give.')
    parser.add_argument('-t', '--trajtest', action='store_true',
                        help='Lets the program stop after generating the starting path.')
    parser.add_argument('-m', '--maxiter', type=int,
                        help='Set the maximum number of iterations.')
    parser.add_argument('-i', '--images', type=int,
                        help='Set the number of images (excluding the end points).')
    return parser.parse_args()


def start_NEB():
    args = parse_args()
    main(args)


if __name__ == '__main__':
    start_NEB()