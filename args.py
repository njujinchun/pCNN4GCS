import argparse
import torch
import json
import random
from pprint import pprint
from utils.misc import mkdirs
import time

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='pCNN with SVGD')        
        self.add_argument('--net', type=str, default='MLRDB', help='Net arch used')
        self.add_argument('--exp-dir', type=str, default="./experiments", help='directory to save experiments')

        # model
        self.add_argument('-ns', '--n-samples', type=int, default=20, help='(20-30) number of model instances in SVGD')
        self.add_argument('--features', type=int, default=48, help='number of basic features in basic conv layers')

        # data      
        self.add_argument('--data-dir', type=str, default="C:/Users/Administrator/Desktop/pCNN/", help='data directory') 
        self.add_argument('--n-train', type=int, default=4000, help="number of training data")
        self.add_argument('--n-test', type=int, default=100, help="number of test data")   
        self.add_argument('--act-fun', type=str, default='PReLU', help="activation function, ReLU, SiLU, PReLU, mish")
                       
        # training
        self.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=0.0025, help='learnign rate')
        self.add_argument('--lr-noise', type=float, default=0.01, help='learnign rate')
        self.add_argument('--batch-size', type=int, default=12, help='batch size for training')
        self.add_argument('--test-batch-size', type=int, default=20, help='batch size for testing')
        self.add_argument('--seed', type=int, default=None, help='manual seed used in Tensor')

        # logging
        self.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')
        self.add_argument('--plot-freq', type=int, default=50, help='how many epochs to wait before plotting test output')
        self.add_argument('--ckpt-freq', type=int, default=100, help='how many epochs to wait before saving model')
        self.add_argument('--ckpt-epoch', type=int, default=None, help='which epoch of checkpoints to be loaded in post mode')

    def parse(self):
        args = self.parse_args()
        date = '/Jul_07'
        args.run_dir = args.exp_dir + date\
            + '/nsamples{}_ntrain{}_nf{}_batch{}_lr{}_noiselr{}_epochs{}_{}_{}'.format(
                args.n_samples, args.n_train, args.features, args.batch_size, args.lr,
                args.lr_noise, args.epochs,args.net, args.act_fun)

        args.ckpt_dir = args.run_dir + '/checkpoints'
        mkdirs([args.run_dir, args.ckpt_dir])
        
        assert args.epochs % args.ckpt_freq == 0, 'epochs must'\
            'be dividable by ckpt_freq'

        # seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        print('Arguments:')
        pprint(vars(args))

        with open(args.run_dir + "/args.txt", 'w') as args_file:
            json.dump(vars(args), args_file, indent=4)

        return args

# global
args = Parser().parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
