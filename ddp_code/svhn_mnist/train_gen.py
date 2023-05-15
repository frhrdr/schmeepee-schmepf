from __future__ import print_function
import sys
sys.path.append('/mnt/d/dp-gfmn/code/')
from test_transfer_learning import load_trained_model, data_loader
from torch.optim.lr_scheduler import StepLR
import sys
import os
from models_gen import FCCondGen, ConvCondGen
# from auxfiles import meddistance
# from full_mmd import mmd_loss
from auxfiles import flatten_features
import torch.optim as optim
import numpy as np
from gen_balanced import log_gen_data, test_results
from gen_balanced import synthesize_data_with_uniform_labels
from synth_data_benchmark import test_gen_data, test_passed_gen_data, datasets_colletion_def
from data_to_feat import data_to_feat
from compress_feat import data_to_feat_compress
from models.model_builder import get_encoders, get_generator, get_mean_and_var_nets
# import sklearn
from sklearn import linear_model
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.utils.data
# sys.path.append('/home/mijungp/dp-gfmn/code/')
from dp_mepf_args import get_args, set_arg_dependencies, get_imagenet_norm_min_and_range
from util import set_random_seed, create_checkpoint, create_synth_dataset, \
  load_checkpoint, get_optimizers
from util_logging import configure_logger, log_losses_and_imgs, route_io_to_file, LOG
from feature_matching import regular_moving_average_update, adam_moving_average_update, \
  extract_and_bound_features, compute_data_embedding
from feature_selection import get_number_of_matching_features
from models.model_builder import get_encoders, get_generator, get_mean_and_var_nets
from torch.utils.tensorboard import SummaryWriter
from dp_functions import dp_dataset_feature_release
from eval_fid import get_fid_scores


# def normalize_data(x_train, x_test):
#     mean = np.mean(x_train)
#     sdev = np.std(x_train)
#     x_train_normed = (x_train - mean) / sdev
#     x_test_normed = (x_test - mean) / sdev
#     assert not np.any(np.isnan(x_train_normed)) and not np.any(np.isnan(x_test_normed))
#
#     return x_train_normed, x_test_normed


def main():

    data_name = 'digits' # 'digits' or 'fashion'
    model_name = 'ResNet'
    domain_adapt = False # if True, we retrain the selected channels on the input layer
    if domain_adapt:
        # which_layer_to_finetune = [1] # first layer
        which_layer_to_finetune = [1, 2] # first and last layers
        # which_layer_to_finetune = [1, 2, 3]  # first two layers
    else:
        which_layer_to_finetune = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device is', device)

    """ 1. Load the trained model and its architecture """
    model2load = 'Trained_ResNet'
    feat_ext = load_trained_model(model2load, model_name, domain_adapt, which_layer_to_finetune, device) # feature extractor

    """ 2. Load data to test """
    batch_size = 100
    train_loader, test_loader = data_loader(batch_size, model_name, data_name)
    train_loader_pub, test_loader_pub = data_loader(batch_size, model_name, 'svhn') # this was for a debugging purpose, or we could use this for compressing the dimension of PFs.

    which_layer_to_use = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] # [17] for last layer only, [1, 17] for top and last layers, or [1:17] for the whole layers
    # which_layer_to_use = [1,17]
    n_matching_layers = len(which_layer_to_use)
    if n_matching_layers == 1: # bottom only
        n_features = feat_ext.n_features_per_layer[1]
    elif n_matching_layers == 2: # bottom and top only
        n_features = feat_ext.n_features_per_layer[1] + feat_ext.n_features_per_layer[-1]
    elif n_matching_layers == 3:  # bottom and top only
        n_features = feat_ext.n_features_per_layer[1:2] + feat_ext.n_features_per_layer[-1]
    elif n_matching_layers==17:
        n_features = sum(feat_ext.n_features_per_layer[1:])
    else:
        print('for now we only support three cases: last layer only, top and last layers, and all layers')

    ### Made sure we get the same results ###
    compression_rate = 100 # keeping top compression_rate % channels
    for batch_idx, (data_pub, labels_pub) in enumerate(train_loader_pub):
        if batch_idx == 0: # just do this for the first batch only, assuming this will be probably similar in other batches.
            data_pub, labels_pub = data_pub.to(device), labels_pub.to(device)
            # out_data_pub, data_feat_pub = feat_ext(data_pub)
            feat_pub, selected_idx = data_to_feat_compress(data_pub, feat_ext, n_matching_layers, which_layer_to_use, compression_rate)
        else:
            break
        print('batch_idx', batch_idx)

    # make sure if the loaded model is correct
    # yes, when svhn was loaded, the accuracy on the batch is nearly 1.
    # print('accuracy on this batch', [torch.mean(1.0*(torch.argmax(out_data_pub, dim=1)==labels_pub)), batch_idx_pub])

    if domain_adapt:
        """ once I know which channels to keep I want to update those on the input layer using the private data """
        optimizer_da = optim.Adam(filter(lambda p: p.requires_grad, feat_ext.parameters()), lr=0.001)
        # but this isn't enough. I also need to mask those that aren't selected. Let's for now post process this.
        criterion = nn.CrossEntropyLoss()

        feat_ext.train()
        domain_adapt_epoch = 1
        for epoch in range(domain_adapt_epoch):  # loop over the dataset multiple times

            running_loss = 0.0

            for batch_idx, (data, labels) in enumerate(train_loader):
                data, labels = data.to(device), labels.to(device)

                optimizer_da.zero_grad()
                y_pred, tmp = feat_ext(data)
                loss_da = criterion(y_pred, labels)

                # if ar.dp_sigma > 0.:
                #     global_norms, global_clips = dp_sgd_backward(classifier.parameters(), loss, device, ar.dp_clip, ar.dp_sigma)
                #     # print(f'max_norm:{torch.max(global_norms).item()}, mean_norm:{torch.mean(global_norms).item()}')
                #     # print(f'mean_clip:{torch.mean(global_clips).item()}')
                # else:
                loss_da.backward()
                optimizer_da.step()
                running_loss += loss_da.item()
                # print()
                # print('mean of conv1 weight values', torch.mean(feat_ext.conv1.weight))
                # print('mean of last layers linear weights', torch.mean(feat_ext.linear.weight))

            # y_pred = feat_ext(test_loader.dataset.data)
            # ROC = roc_auc_score(test_loader.dataset.targets, y_pred.detach().numpy())
            #     print('Epoch {}: loss : {}'.format(epoch, loss))

        print('Finished domain adaptation')

        # once done training the top layer, we freeze all the parameters in feat_ext
        for param in feat_ext.parameters():
            param.requires_grad = False

    print('num of features before pruning', n_features)
    n_features = feat_pub.shape[1]
    print('num of features after pruning', n_features)


    """ 3. Define a generator """
    input_size = 5 # dimension of z
    n_classes = 10
    n_epochs = 50
    model = ConvCondGen(input_size, '500,500', n_classes, '16,8', '5,5', use_sigmoid=True, batch_norm=True).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    ### Computing the mean embedding of private data distribution ###
    print("Computing the mean embedding of private data distribution, this will take a while!")
    data_embedding = torch.zeros(2*n_features, n_classes, device=device)
    n_priv_data_samps = train_loader.dataset.data.shape[0]

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        for idx_class in range(n_classes):
            idx_data = data[labels == idx_class]
            if idx_data.shape[0]==0:
                feat_priv_data = torch.zeros(1)
            else:
                feat_priv_data = data_to_feat(idx_data, feat_ext, n_matching_layers, which_layer_to_use, selected_idx=selected_idx)
            data_embedding[0:n_features, idx_class] += torch.sum(feat_priv_data, dim=0)
            data_embedding[n_features:, idx_class] += torch.sum(feat_priv_data**2, dim=0)

    data_embedding = data_embedding / n_priv_data_samps

    print("Now start training a generator!")

    log_dir = 'logs/gen/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for epoch in range(1, n_epochs + 1):
        for batch_idx, (data, labels) in enumerate(train_loader):
            # data, labels = data.to(device), labels.to(device)

            gen_code, gen_labels = model.get_code(batch_size, device)
            gen_samples = model(gen_code) # batch_size by 784

            syn_data_embedding = torch.zeros(2 * n_features, n_classes, device=device)
            _, gen_labels_numerical = torch.max(gen_labels, dim=1)
            for idx_class in range(n_classes):
                idx_gen_data = gen_samples[gen_labels_numerical == idx_class]
                if idx_gen_data.shape[0]==0:
                    feat_syn_data = torch.zeros(1)
                else:
                    idx_gen_data = torch.reshape(idx_gen_data, (idx_gen_data.shape[0], 1, 28, 28))
                    feat_syn_data = data_to_feat(idx_gen_data.repeat((1,3,1,1)), feat_ext, n_matching_layers, which_layer_to_use, selected_idx=selected_idx)
                syn_data_embedding[0:n_features, idx_class] += torch.sum(feat_syn_data, dim=0)
                syn_data_embedding[n_features:, idx_class] += torch.sum(feat_syn_data ** 2, dim=0)

            syn_data_embedding = syn_data_embedding / batch_size

            loss = torch.sum(torch.abs(data_embedding - syn_data_embedding))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # at every epoch, we print this
        print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch, loss.item()))
        log_gen_data(model, device, epoch, n_classes, log_dir)
        scheduler.step()

    # evaluating synthetic data on a classifier
    syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, gen_batch_size=batch_size,
                                                               n_data=n_priv_data_samps,
                                                               n_labels=n_classes)

    dir_syn_data = log_dir + data_name + '/synthetic_mnist'
    if not os.path.exists(dir_syn_data):
        os.makedirs(dir_syn_data)

    np.savez(dir_syn_data, data=syn_data, labels=syn_labels)

    # normalize data
    test_data = test_loader.dataset.data.view(test_loader.dataset.data.shape[0], -1).numpy()
    # syn_data, test_data = normalize_data(syn_data, test_data)

    LR = linear_model.LogisticRegression(solver='lbfgs', max_iter=50000, multi_class = 'auto')
    LR.fit(syn_data, syn_labels.squeeze())
    y_pred = LR.predict(test_data)
    acc = accuracy_score(y_pred, test_loader.dataset.targets.numpy())
    print('on logistic regression, accuracy is', acc)

if __name__ == '__main__':
    main()
