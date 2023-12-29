import h5py
import torch as th
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score
from sklearn import preprocessing
import pandas as pd
import scipy.io
plt.switch_backend('agg')  


def load_data(hdf5_dir, args, kwargs, flag):
    with h5py.File(hdf5_dir + "/input.hdf5", 'r') as f:
        x = f['dataset'][()]           
    x = np.expand_dims(x,axis=1) 
    
    with h5py.File(hdf5_dir + "/output.hdf5", 'r') as f:
        img = f['dataset'][()]            
      
    if flag == 'train':
        x = x[:args.n_train]
        img = img[:args.n_train]
        batch_size = args.batch_size
    elif flag == 'test':
        x = x[-args.n_test:] 
        img = img[-args.n_test:]      
        batch_size = args.test_batch_size   

    y_mean = np.mean(img, 0)
    y_var = np.sum((img - np.mean(img, 0)) ** 2)  
    data = th.utils.data.TensorDataset(th.FloatTensor(x), th.FloatTensor(img))
    data_loader = th.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    n_out_pixels = np.prod(img.shape) 
    
    return x, img, n_out_pixels, y_mean, y_var, data_loader

def plot_pred(x, y, y_hat, epoch, idx, output_dir):
    y=y[0,:,:,:]    
    y_hat=y_hat[0,:,:,:]
    x=x[0,:,:,:]    
    error = y - y_hat
    print(x.shape[0])
    samples = np.row_stack((x, y, y_hat, error))  
    Nout = y.shape[0]                             
    print(samples.shape)
    print(Nout)
    c_max = np.full( (Nout*3), 0.0)
    c_min = np.full( (Nout*3), 0.0)
    for l in range(Nout*3):
        if l < 2*Nout:
            c_max[l] = np.max(samples[l+Nout])
    
        else:
            c_max[l] = np.max( np.abs(samples[l+Nout]) )
            c_min[l] = 0. - np.max( np.abs(samples[l+Nout]) )

    title = (['$Layer1$','$Layer2$','$Layer3$','$Layer4$','$Layer5$','$Layer6$','$Layer7$','$Layer8$','$Layer9$','$Layer10$','$Rate$'])
    ylabel = (['$\mathbf{k}$','$\mathbf{S}_g$', '$\hat{\mathbf{S}}_g$', '$\mathbf{S}_g-\hat{\mathbf{S}}_g$'])
    fs = 12
    fig, axes = plt.subplots(4, Nout, figsize=(Nout*4,10))   
    k = 0
    for j, ax in enumerate(fig.axes):
        print(j)
        if j < Nout:
            cax = ax.imshow(samples[j], cmap='jet', origin='lower')
        else:
            cax = ax.imshow(samples[j], cmap='jet', origin='lower',vmin=c_min[j-Nout], vmax=c_max[j-Nout])
        cbar = plt.colorbar(cax, ax=ax,fraction=0.024, pad=0.015)
        cbar.ax.tick_params(axis='both', which='both', length=0)
        cbar.ax.tick_params(labelsize=fs-2)
        if j < Nout:
            ax.set_title(title[j],fontsize=fs)

        if j%Nout == 0:
            ax.set_ylabel(ylabel[k], fontsize=fs+2)
            k = k + 1

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['left'].set_color('white')   
        ax.spines['right'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')

    plt.savefig(output_dir + '/epoch_{}_output_{}.png'.format(epoch,idx), bbox_inches='tight',dpi=400)
    plt.close(fig)     
    print("epoch {}, done with printing sample output {}".format(epoch, idx))


def plot_r2_rmse(r2_train, r2_test, rmse_train, rmse_test, exp_dir, args):
    x = np.arange(args.log_interval, args.n_epochs + args.log_interval,
                args.log_interval)  
    plt.figure()
    plt.plot(x, r2_train, 'k', label="train: {:.3f}".format(np.mean(r2_train[-5: -1])))
    plt.plot(x, r2_test, 'r', linestyle = '--', label="test: {:.3f}".format(np.mean(r2_test[-5: -1])))
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('$R^2$', fontsize=14)
    plt.legend(loc='lower right')
    plt.savefig(exp_dir + "/r2.png", dpi=400)
    plt.close()
    np.savetxt(exp_dir + "/r2_train.txt", r2_train)
    np.savetxt(exp_dir + "/r2_test.txt", r2_test)

    plt.figure()
    plt.plot(x, rmse_train, 'k', label="train: {:.3f}".format(np.mean(rmse_train[-5: -1])))
    plt.plot(x, rmse_test, 'r', linestyle = '--', label="test: {:.3f}".format(np.mean(rmse_test[-5: -1])))
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.legend(loc='upper right')
    plt.savefig(exp_dir + "/rmse.png", dpi=400)
    plt.close()
    np.savetxt(exp_dir + "/rmse_train.txt", rmse_train)
    np.savetxt(exp_dir + "/rmse_test.txt", rmse_test)
  