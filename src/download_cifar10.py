import torchvision.datasets as dset
import argparse

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
args = parser.parse_args()

def main():
  train_data = dset.CIFAR10(root=args.data, train=True, download=True)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True)


if __name__ == '__main__':
  main() 
