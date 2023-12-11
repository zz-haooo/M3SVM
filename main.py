import numpy as np
import argparse
from torch.optim import Adam
import matplotlib.pyplot as plt
import scipy.io as scio

from train import *


if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, default='vehicle_uni')
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=int, default=0.000)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--state', type=int, default=42)
    parser.add_argument('--If_scale', default=True)
    para = parser.parse_args()
    para.test_size = 0.2

    para.data = 'HHAR'

    if para.data == 'HHAR':  # n=10229,d=561,c=6
        para.lr = 0.02
        para.lam = 0.0005
        para.p = 4
    elif para.data == 'Cornell':  # n=827,d=4134,c=7
        para.lr = 0.1
        para.lam = 0.005
        para.p = 1
        para.If_scale = False
    elif para.data == 'USPS':  # n=9298,d=256,c=10
        para.lr = 0.01
        para.lam = 0.001
        para.p = 4
    elif para.data == 'ISOLET':  # n=1560,d=617,c=26
        para.lr = 0.001
        para.lam = 0.001
        para.p = 4
        para.If_scale = False
    elif para.data == 'ORL':  # n=400,d=1024,c=40
        para.lr = 0.01
        para.lam = 0.01
        para.p = 6
    elif para.data == 'Dermatology':  # n=366,d=34,c=6
        para.lr = 0.01
        para.lam = 0.1
        para.p = 6
        para.If_scale = False
    elif para.data == 'Vehicle':  # n=946,d=18,c=4
        para.lr = 0.05
        para.lam = 0.0001
        para.p = 4
    elif para.data == 'Glass':  # n=214,d=9,c=6
        para.lr = 0.01
        para.lam = 0.0001
        para.p = 4

    R_MLR(para)