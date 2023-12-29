
from mlrdb_mish import MLRDB
import torch 
import numpy as np
import torch.nn as nn
import h5py
import os
import sys
import json
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from time import time
from sklearn.metrics import r2_score
import scipy.io
plt.switch_backend('agg')   
import warnings
import seaborn as sns
warnings.filterwarnings("ignore") 
from args import args
from models.bayes_nn import BayesNN
from models.svgd import SVGD
from utils.misc import mkdirs, logger
from utils.plot import save_stats 
from utils.data_plot_cnn_fc import load_data, plot_pred, plot_r2_rmse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
kwargs = {'num_workers': 4,'pin_memory': True} if torch.cuda.is_available() else {}

args.train_dir = args.run_dir + "/training"
args.pred_dir = args.run_dir + "/predictions"
mkdirs([args.train_dir, args.pred_dir])

print('Loaded data!')
hdf5_dir = args.data_dir + "data"
x_train, img_train, n_out_pixels_train, y_train_mean, y_train_var, train_loader = load_data(hdf5_dir, args, kwargs, 'train')
x_test, img_test, n_out_pixels_test, y_test_mean, y_test_var, test_loader = load_data(hdf5_dir, args, kwargs, 'test')
# simple statistics of training and testing output data    
train_stats = {}
train_stats['y_mean'] = y_train_mean
train_stats['y_var'] = y_train_var  
test_stats = {}
test_stats['y_mean'] = y_test_mean
test_stats['y_var'] = y_test_var
logger['train_output_var'] = train_stats['y_var']
logger['test_output_var'] = test_stats['y_var']

if args.net == 'MLRDB':    
    model = MLRDB(x_train.shape[1], img_train.shape[1],filters=args.features).to(device)
print(model)
print("number of parameters: {}/nnumber of layers: {}".format(*model._num_parameters_convlayers()))

bayes_nn = BayesNN(model, n_samples=args.n_samples).to(device)

svgd = SVGD(bayes_nn, train_loader)

def test(epoch, logger): 
    bayes_nn.eval()
    
    mse_test, nlp_test = 0., 0.
    for batch_idx, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        mse, nlp, output = bayes_nn._compute_mse_nlp(input, target, 
                            size_average=True, out=True)
        
        y_pred_mean = output.mean(0)        
        EyyT = (output ** 2).mean(0)
        EyEyT = y_pred_mean ** 2
        y_noise_var = (- bayes_nn.log_beta).exp().mean()
        y_pred_var =  EyyT - EyEyT + y_noise_var

        mse_test += mse.item()
        nlp_test += nlp.item()

    rmse_test = np.sqrt(mse_test / len(test_loader))
    r2_test = 1 - mse_test * target.numel() / logger['test_output_var']
    mnlp_test = nlp_test / len(test_loader)    
    logger['rmse_test'].append(rmse_test)
    logger['r2_test'].append(r2_test)
    logger['mnlp_test'].append(mnlp_test)
    print("epoch {}, testing  r2: {:.4f}, test rmse: {}".format(epoch, r2_test, rmse_test))

def pred_samples(x):
    bayes_nn.eval()
    y_pred, y_pred_var = np.zeros_like(img_test), np.zeros_like(img_test)    
    for i in range(x.shape[0]):
        x_tensor = (torch.FloatTensor(x[[i]])).to(device)
        with torch.no_grad():
            y_hat, pred_var = bayes_nn.predict(x_tensor)
        y_hat, pred_var = y_hat.data.cpu().numpy(), pred_var.data.cpu().numpy()    
        y_pred[i], y_pred_var[i] = y_hat, pred_var  
   
print('Start training.........................................................')
tic = time()
for epoch in range(1, args.epochs + 1):    
    svgd.train(epoch, logger)
    with torch.no_grad():
        test(epoch, logger)
training_time = time() - tic
print('Finished training:\n{} epochs\n{} samples (SVGD)\n{} seconds'
    .format(args.epochs, args.n_samples, training_time))

x_axis = np.arange(args.log_freq, args.epochs + args.log_freq, args.log_freq) 
save_stats(args.train_dir, logger, x_axis)

args.training_time = training_time
args.n_params, args.n_layers = model._num_parameters_convlayers()
with open(args.run_dir + "/args.txt", 'w') as args_file:
    json.dump(vars(args), args_file, indent=4)
