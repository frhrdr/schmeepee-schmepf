from __future__ import print_function
import sys

from test_transfer_learning import load_trained_model, data_loader
from torch.optim.lr_scheduler import StepLR
import sys
import os
from models_gen import FCCondGen, ConvCondGen
import torch.optim as optim
import numpy as np
from gen_balanced import log_gen_data, test_results
from gen_balanced import synthesize_data_with_uniform_labels
from auxfiles import log_args
# from synth_data_benchmark import test_gen_data, test_passed_gen_data, datasets_colletion_def
from downstream_models import test_gen_data, log_final_score
from data_to_feat import data_to_feat
from compress_feat import data_to_feat_compress
# from models.model_builder import get_encoders, get_generator, get_mean_and_var_nets
# import sklearn
from sklearn import linear_model
from sklearn.metrics import accuracy_score

from torch.nn.parallel import DistributedDataParallel as DDP

import torch
import torch.nn as nn
import torch.utils.data
# sys.path.append('/home/mijungp/dp-gfmn/code/')
sys.path.append('/mnt/d/dp-gfmn/code/')

# from util import LOG, get_optimizers
# from models.model_builder import ConstMat
import argparse
from collections import namedtuple
import colorlog
LOG = colorlog.getLogger(__name__)

import math
from sklearn.model_selection import ParameterGrid


class ConstMat(nn.Module):  # hacky way to get DDP to work with these single parameters
  def __init__(self, in_features, out_features, device, dtype=None):
    super(ConstMat, self).__init__()
    self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

  def forward(self, _):
    return self.weight

  def fwd(self):
    return self.forward(None)

def get_optimizers(net_gen, gen_lr, net_mean, net_var, m_avg_lr, beta1, ckpt):
  optimizer_gen = optim.Adam(net_gen.parameters(), lr=gen_lr, betas=(beta1, 0.999))
  if ckpt is not None:
    optimizer_gen.load_state_dict(ckpt['opt_gen'])

  if net_mean is not None and net_var is not None:
    optimizer_mean = optim.Adam(net_mean.parameters(), lr=m_avg_lr, betas=(beta1, 0.999))
    optimizer_var = optim.Adam(net_var.parameters(), lr=m_avg_lr, betas=(beta1, 0.999))
    if ckpt is not None:
      optimizer_mean.load_state_dict(ckpt['opt_mean'])
      optimizer_var.load_state_dict(ckpt['opt_var'])
  else:
    optimizer_mean, optimizer_var = None, None

  opt_tuple = namedtuple('optimizers', ['gen', 'mean', 'var'])
  return opt_tuple(optimizer_gen, optimizer_mean, optimizer_var)



def get_args():

    parser = argparse.ArgumentParser()

    # BASICS
    #parser.add_argument('--seed', type=int, default=None, help='sets random seed')
    parser.add_argument('--base-log-dir', type=str, default='logs/gen/', help='path where logs for all runs are stored')
    # parser.add_argument('--log-name', type=str, default=None, help='subdirectory for this run')
    # parser.add_argument('--log-dir', type=str, default=None, help='override save path. constructed if None')
    parser.add_argument('--data-name', type=str, default='digits', help='options are digits and fashion')


    # OPTIMIZATION
    parser.add_argument('--batch-size', '-bs', type=int, default=100)
    parser.add_argument('--epochs', '-ep', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate, default=0.0002')

    parser.add_argument('--loss-type', type=str, default='l2', help='either l2 or l1 loss')
    # these two below only matters if loss-type == l2
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam optimizer')
    # parser.add_argument('--lr-decay', type=float, default=0.9, help='per epoch learning rate decay factor')
    parser.add_argument('--m_avg_lr', type=float, default=1e-5,
                        help='Learning rate for moving average, default=0.0002')

    # ALTERNATE MODES
    parser.add_argument('--mean-only', default=False, help='mean only or mean and variance both for features')
    parser.add_argument('--domain-adapt', action='store_true', default=False,
                        help='do domain adaptation, if true')
    # if domain-adapt is true, you want to specify which layers you want to fine-tune with private data
    # parser.add_argument('--which-layer-domain-adapt', type=str, default='1,2,3', help='specify which layers you want to fine-tune with private data')

    # parser.add_argument('--which-layer-to-use', type=int,
    #                     default=[1, 17],
    #                     help='which layers of features to use as a list of integers')

    parser.add_argument('--which-layer-to-use', type=int, default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], help='which layers of features to use as a list of integers')
    parser.add_argument('--feat-selection-perc', type=int, default=100, help='for selecting how many features to use from each layer')

    # DP SPEC
    parser.add_argument('--is-private', default=False, help='produces a DP mean embedding of data')
    parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon in (epsilon, delta)-DP')
    parser.add_argument('--delta', type=float, default=1e-5, help='delta in (epsilon, delta)-DP')

    ar = parser.parse_args()

    # preprocess_args(ar)
    # log_args(ar.log_dir, ar)
    return ar

# def preprocess_args(ar):
#     # if ar.log_dir is None:
#     #     assert ar.log_name is not None
#     #     ar.log_dir = ar.base_log_dir + ar.log_name + '/'
#     # if not os.path.exists(ar.log_dir):
#     #     os.makedirs(ar.log_dir)
#
#     if ar.seed is None:
#         ar.seed = np.random.randint(0, 1000)
#
#     if ar.seed is None:
#         ar.seed = np.random.randint(0, 1000)
#         assert ar.data in {'digits', 'fashion'}

def get_mean_and_var_nets_classification(n_features_in_enc, n_classes, device, ckpt, ddp_rank):

  net_mean = ConstMat(n_features_in_enc, n_classes, device)
  net_var = ConstMat(n_features_in_enc, n_classes, device)

  if ddp_rank is not None:
    net_mean = DDP(net_mean, device_ids=[ddp_rank])
    net_var = DDP(net_var, device_ids=[ddp_rank])
  if ckpt is not None:
    net_mean.load_state_dict(ckpt['net_mean'])
    net_var.load_state_dict(ckpt['net_var'])
  LOG.debug(net_mean)
  LOG.debug(net_var)

  return net_mean, net_var


def adam_ma(fake_data_feat_mean, fake_data_feat_var, global_feat_mean_values, global_feat_var_values,
                               optimizer_g, criterion_l2_loss,
                               avg_loss_net_gen_mean, net_mean, avg_loss_net_mean, optimizer_mean,
                               avg_loss_net_gen_var, net_var, avg_loss_net_var, optimizer_var, mean_only):

  def match_loss(moment_net, fake_moment, real_moment, avg_loss_net_mom, opt_moment,
                 avg_loss_net_gen):
    moment_diff_m_avg = moment_net(None).T # num features by num_classes
    diff = real_moment.detach() - fake_moment.detach()
    loss_net_moment = criterion_l2_loss(moment_diff_m_avg, diff.detach())

    avg_loss_net_mom += loss_net_moment.item()
    loss_net_moment.backward()
    opt_moment.step()  # update moment net

    diff_true = torch.sum(torch.reshape(moment_diff_m_avg, (-1,)) * torch.reshape(real_moment, (-1,))).detach()
    diff_fake = torch.sum(torch.reshape(moment_diff_m_avg, (-1,)) * torch.reshape(fake_moment, (-1,)))

    loss_net_gen = (diff_true - diff_fake)  # compute loss for generator
    avg_loss_net_gen += loss_net_gen.item()

    return avg_loss_net_mom, avg_loss_net_gen, loss_net_gen

  res_mean = match_loss(net_mean, fake_data_feat_mean, global_feat_mean_values,
                        avg_loss_net_mean, optimizer_mean, avg_loss_net_gen_mean)
  avg_loss_net_mean, avg_loss_net_gen_mean, loss_net_g_mean = res_mean

  res_var = match_loss(net_var, fake_data_feat_var, global_feat_var_values,
                         avg_loss_net_var, optimizer_var, avg_loss_net_gen_var)
  avg_loss_net_var, avg_loss_net_gen_var, loss_net_g_var = res_var
  if mean_only:
    loss_net_g = loss_net_g_mean
  else:
    loss_net_g = loss_net_g_mean + loss_net_g_var
  loss_net_g.backward()
  optimizer_g.step()

  return avg_loss_net_gen_mean, avg_loss_net_mean, avg_loss_net_gen_var, avg_loss_net_var



def main(single_run, data_name, seed, lr, loss_type, batch_size, compression_rate, mean_only, n_epochs, m_avg_lr, beta1, is_private, epsilon, delta):
    ar = get_args()
    # print(ar)
    # log_dir = ar.log_dir
    domain_adapt = ar.domain_adapt # assuming this is always false
    which_layer_to_use = ar.which_layer_to_use

    if single_run:
        # unpack the arguments from ar
        data_name = ar.data_name
        seed = np.random.randint(0, 1000)
        lr = ar.lr
        loss_type = ar.loss_type
        batch_size = ar.batch_size
        compression_rate = ar.feat_selection_perc
        n_epochs = ar.epochs
        mean_only = ar.mean_only
        m_avg_lr = ar.m_avg_lr
        beta1 = ar.beta1
        is_private = ar.is_private
        epsilon = ar.epsilon
        delta = ar.delta
    # else: # we use what's given in the main function arguments


    log_name =  'data=' + str(data_name) + '_' + 'seed=' + str(seed) + '_' + \
        'lr=' + str(lr) + '_' + 'loss_type=' + str(loss_type) + '_' + \
        'batch_size=' + str(batch_size) + '_' + 'compression_rate=' + str(compression_rate) + '_' +  'mean_only=' + str(mean_only) + '_' + \
        'n_epochs=' + str(n_epochs) + '_' + 'm_avg_lr=' + str(m_avg_lr) + '_' + 'beta1=' + str(beta1) + '_' + \
        'is_private=' + str(is_private) + '_' + 'epsilon=' + str(epsilon) + '_' + 'delta=' + str(delta)
    print('log_name is', log_name)

    log_dir = ar.base_log_dir + log_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    model_name = 'ResNet'
    if domain_adapt:
        # which_layer_to_finetune = [1] # first layer
        which_layer_to_finetune = [1, 2]  # first and last layers
        # which_layer_to_finetune = [1, 2, 3]  # first two layers
    else:
        which_layer_to_finetune = []


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device is', device)

    """ 1. Load the trained model and its architecture """
    if data_name == 'digits':
        model2load = 'Trained_ResNet'
    else: # fashion
        print('loading pretrained resnet with cifar10')
        model2load = 'Trained_ResNet_cifar10'
    feat_ext = load_trained_model(model2load, model_name, domain_adapt, which_layer_to_finetune,
                                  device)  # feature extractor

    """ 2. Load data to test """
    train_loader, test_loader = data_loader(batch_size, model_name, data_name)

    if data_name == 'digits':
        train_loader_pub, test_loader_pub = data_loader(batch_size, model_name, 'svhn')  # this was for a debugging purpose, or we could use this for compressing the dimension of PFs.
    else: # fashion
        train_loader_pub, test_loader_pub = data_loader(batch_size, model_name,
                                                        'cifar10')  # this was for a debugging purpose, or we could use this for compressing the dimension of PFs.


    n_matching_layers = len(which_layer_to_use)
    n_features = sum(feat_ext.n_features_per_layer[1:])

    ### Made sure we get the same results ###
    # compression_rate = 100  # keeping top compression_rate % channels

    for batch_idx, (data_pub, labels_pub) in enumerate(train_loader_pub):
        if batch_idx == 0:  # just do this for the first batch only, assuming this will be probably similar in other batches.
            data_pub, labels_pub = data_pub.to(device), labels_pub.to(device)
            feat_pub, selected_idx = data_to_feat_compress(data_pub, feat_ext, n_matching_layers, which_layer_to_use,
                                                           compression_rate)
        else:
            break
        print('batch_idx', batch_idx)

    # if domain_adapt:
    #     """ once I know which channels to keep I want to update those on the input layer using the private data """
    #     optimizer_da = optim.Adam(filter(lambda p: p.requires_grad, feat_ext.parameters()), lr=0.001)
    #     criterion = nn.CrossEntropyLoss()
    #
    #     feat_ext.train()
    #     domain_adapt_epoch = 1
    #     for epoch in range(domain_adapt_epoch):  # loop over the dataset multiple times
    #
    #         running_loss = 0.0
    #
    #         for batch_idx, (data, labels) in enumerate(train_loader):
    #             data, labels = data.to(device), labels.to(device)
    #
    #             optimizer_da.zero_grad()
    #             y_pred, tmp = feat_ext(data)
    #             loss_da = criterion(y_pred, labels)
    #
    #             """ this is for later, for testing DP settings """
    #             # if ar.dp_sigma > 0.:
    #             #     global_norms, global_clips = dp_sgd_backward(classifier.parameters(), loss, device, ar.dp_clip, ar.dp_sigma)
    #             #     # print(f'max_norm:{torch.max(global_norms).item()}, mean_norm:{torch.mean(global_norms).item()}')
    #             #     # print(f'mean_clip:{torch.mean(global_clips).item()}')
    #             # else:
    #             loss_da.backward()
    #             optimizer_da.step()
    #             running_loss += loss_da.item()
    #
    #     print('Finished domain adaptation')
    #
    #     # once done training the top layer, we freeze all the parameters in feat_ext
    #     for param in feat_ext.parameters():
    #         param.requires_grad = False

    print('num of features before pruning', n_features)
    n_features = feat_pub.shape[1]
    print('num of features after pruning', n_features)

    """ 3. Define a generator : this is a very specific to MNIST image size data """
    input_size = 5  # dimension of z
    n_classes = 10
    net_gen = ConvCondGen(input_size, '500,500', n_classes, '16,8', '5,5', use_sigmoid=True, batch_norm=True).to(device)
    # lr = ar.lr

    if loss_type == 'l2':
        net_mean, net_var = get_mean_and_var_nets_classification(n_features, n_classes, device, ckpt=None, ddp_rank=None) # data independent initialization for
        """ these are the default hyperparameter settings from dp_gfmn_"""
        optimizers = get_optimizers(net_gen, lr, net_mean, net_var, m_avg_lr, beta1, ckpt=None)
    else: # l1 loss
        optimizer = torch.optim.Adam(list(net_gen.parameters()), lr=lr)

    # scheduler = StepLR(optimizers.gen, step_size=1, gamma=0.9)

    ### Computing the mean embedding of private data distribution ###
    print("Computing the mean embedding of private data distribution, this will take a while!")
    data_embedding = torch.zeros(2 * n_features, n_classes, device=device)
    n_priv_data_samps = train_loader.dataset.data.shape[0]

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        for idx_class in range(n_classes):
            idx_data = data[labels == idx_class]
            if idx_data.shape[0] == 0:
                feat_priv_data = torch.zeros(1)
            else:
                feat_priv_data = data_to_feat(idx_data, feat_ext, n_matching_layers, which_layer_to_use,
                                              selected_idx=selected_idx)
            data_embedding[0:n_features, idx_class] += torch.sum(feat_priv_data, dim=0)
            data_embedding[n_features:, idx_class] += torch.sum(feat_priv_data ** 2, dim=0)

    data_embedding = data_embedding / n_priv_data_samps
    real_feat_means = data_embedding[0:n_features, :]
    real_feat_vars = data_embedding[n_features:, :]

    print("Now start training a generator!")

    # log_dir = 'logs/gen/'


    if loss_type == 'l2':
        avg_loss_net_gen_mean = 0.0
        avg_loss_net_gen_var = 0.0
        avg_loss_net_mean = 0.0
        avg_loss_net_var = 0.0

        criterion_loss = nn.MSELoss()  # or we change it to abolute loss
    # criterion_loss = nn.L1Loss()

    for epoch in range(1, n_epochs + 1):
        for batch_idx, (data, labels) in enumerate(train_loader):
            # data, labels = data.to(device), labels.to(device)

            gen_code, gen_labels = net_gen.get_code(batch_size, device)
            gen_samples = net_gen(gen_code)  # batch_size by 784

            syn_data_embedding = torch.zeros(2 * n_features, n_classes, device=device)
            _, gen_labels_numerical = torch.max(gen_labels, dim=1)
            for idx_class in range(n_classes):
                idx_gen_data = gen_samples[gen_labels_numerical == idx_class]
                if idx_gen_data.shape[0] == 0:
                    feat_syn_data = torch.zeros(1)
                else:
                    idx_gen_data = torch.reshape(idx_gen_data, (idx_gen_data.shape[0], 1, 28, 28))
                    feat_syn_data = data_to_feat(idx_gen_data.repeat((1, 3, 1, 1)), feat_ext, n_matching_layers,
                                                 which_layer_to_use, selected_idx=selected_idx)
                syn_data_embedding[0:n_features, idx_class] += torch.sum(feat_syn_data, dim=0)
                syn_data_embedding[n_features:, idx_class] += torch.sum(feat_syn_data ** 2, dim=0)

            syn_data_embedding = syn_data_embedding / batch_size

            if loss_type == 'l1':
                if mean_only: # if mean_only is true, then we use only the mean for features
                    loss = torch.sum(torch.abs(data_embedding - syn_data_embedding))
                else: # we use both mean and variance
                    loss = torch.sum(torch.abs(data_embedding[0:n_features,:] - syn_data_embedding[0:n_features,:]))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else: #l2
                fake_data_feat_mean = syn_data_embedding[0:n_features, :]  # how to handle the several classes?
                fake_data_feat_var = syn_data_embedding[n_features:, :]  # how to handle the several classes?

                net_gen.zero_grad(set_to_none=True)
                net_mean.zero_grad(set_to_none=True)
                net_var.zero_grad(set_to_none=True)
                new_vals = adam_ma(fake_data_feat_mean, fake_data_feat_var, real_feat_means,
                                                      real_feat_vars, optimizers.gen, criterion_loss,
                                                      avg_loss_net_gen_mean, net_mean,
                                                      avg_loss_net_mean, optimizers.mean,
                                                      avg_loss_net_gen_var, net_var,
                                                      avg_loss_net_var, optimizers.var, mean_only)
                avg_loss_net_gen_mean, avg_loss_net_mean, avg_loss_net_gen_var, avg_loss_net_var = new_vals

        # at every epoch, we print this
        print('Train Epoch: {}'.format(epoch))
        log_gen_data(net_gen, device, epoch, n_classes, log_dir)
        # scheduler.step()

    # evaluating synthetic data on a classifier
    syn_data, syn_labels = synthesize_data_with_uniform_labels(net_gen, device, gen_batch_size=batch_size,
                                                               n_data=n_priv_data_samps,
                                                               n_labels=n_classes)

    dir_syn_data = log_dir + data_name + '/synthetic_mnist'
    if not os.path.exists(dir_syn_data):
        os.makedirs(dir_syn_data)

    np.savez(dir_syn_data, data=syn_data, labels=syn_labels)

    # normalize data
    # test_data = test_loader.dataset.data.view(test_loader.dataset.data.shape[0], -1).numpy()
    # syn_data, test_data = normalize_data(syn_data, test_data)
    final_score = test_gen_data(data_log_name=log_name + '/' + data_name, data_key=data_name, subsample=1.0)
    log_final_score(log_dir, final_score)

    return final_score[-1]

if __name__ == '__main__':

    ##### Make sure you're running just a single run or doing grid search (then single_run is False) ####
    single_run = False

    if single_run:
        avg_final_score = main(single_run, [], [], [], [], [], [], [], [], [], [], [], [], [])
    else:
        # (single_run, data_name, seed, lr, loss_type, batch_size, compression_rate, mean_only, n_epochs, m_avg_lr, beta1,
         # is_private, epsilon, delta)
        # seed_num_arr = [1, 2, 3, 4, 5]
        # data_arr = ['digits', 'fashion']
        # data_arr = ['fashion']
        data_arr = ['digits']
        epoch_arr = [50, 100, 200, 400]
        batch_size_arr = [100, 200, 400]
        # lr_arr = [5e-5, 1e-3]
        lr_arr = [5e-5]
	    # loss_type_arr = ['l1', 'l2']
        loss_type_arr = ['l2']
        beta1_arr = [0.2, 0.5, 0.7]
        m_avg_lr_arr = [1e-5, 0.0002, 1e-3]
        mean_only_arr = ['True', 'False']
        feat_selection_perc_arr = [10, 20, 50, 80, 100]
        is_private = ['False']
        epsilon = [1.0]
        delta = [1e-5]

        grid = ParameterGrid({"data_name":data_arr, "lr":lr_arr, "loss_type":loss_type_arr, "batch_size":batch_size_arr, "compression_rate":feat_selection_perc_arr, "mean_only":mean_only_arr, "n_epochs":epoch_arr, "m_avg_lr":m_avg_lr_arr, "beta1":beta1_arr,
         "is_private":is_private, "epsilon":epsilon, "delta":delta})

        repetitions = 5

        for elem in grid:
            print(elem, "\n")
            scr_all = []
            for ii in range(repetitions):
                scr = main(single_run, elem["data_name"], ii, elem["lr"], elem["loss_type"], elem["batch_size"], elem["compression_rate"], elem["mean_only"], elem["n_epochs"],
                       elem["m_avg_lr"], elem["beta1"], elem["is_private"], elem["epsilon"], elem["delta"])
                scr_all.append(scr)

            print('Average accuray across 5 runs is', np.mean(scr_all))










































