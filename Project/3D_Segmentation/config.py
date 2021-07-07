import argparse


parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=0,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data in/out and dataset
parser.add_argument('--dataset_path', default='./fixed',
                    help='fixed trainset root path')
parser.add_argument('--test_dataset_path', default='./test',
                    help='fixed testset root path')

parser.add_argument('--save', default='3d-unet',
                    help='save path of trained model')

parser.add_argument('--train_resize_scale', type=float,
                    default=1.0, help='resize scale for input data')
parser.add_argument('--test_resize_scale', type=float,
                    default=1.0, help='resize scale for input data')

parser.add_argument('--crop_size', type=list,
                    default=[32, 64, 64], help='patch size of train samples after resize')

parser.add_argument('--batch_size', type=list, default=6,
                    help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=200,
                    metavar='N', help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.0001,
                    metavar='LR', help='learning rate (default: 0.01)')

parser.add_argument('--early-stop', default=20, type=int,
                    help='early stopping (default: 20)')

parser.add_argument('--n_labels', type=int, default=2,
                    help='number of classes')
args = parser.parse_args()
