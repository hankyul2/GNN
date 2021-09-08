import argparse

from src.train import run

parser = argparse.ArgumentParser(description='GNN pytorch repo')
parser.add_argument('-g', '--gpu_id', type=str, default='cpu', help='enter gpu number')


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)