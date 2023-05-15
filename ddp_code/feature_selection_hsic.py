import math
import pyHSICLasso as hsic
import numpy as np
import torch as pt
from models.encoders_class import Encoders
from collections import defaultdict
from util_logging import LOG


def collect_feats_and_labels(encoders, data_loader, device, n_batches):
  layer_feats_dict = defaultdict(list)
  labels_list = []
  for idx, batch in enumerate(data_loader):
    if idx >= n_batches:
      break

    x, y = batch
    encoders.load_features(x.to(device))
    labels_list.append(y)
    for layer_name, layer_act in encoders.layer_feats.items():
      layer_feats_dict[layer_name].append(layer_act.cpu())

  layer_feats_cat_dict = dict()
  labels = pt.cat(labels_list)
  for layer_name, layer_act_list in layer_feats_dict.items():
    layer_feats_cat_dict[layer_name] = pt.cat(layer_act_list)

  return layer_feats_cat_dict, labels


def get_layer_names(encoders, data_loader, device):
  for x, _ in data_loader:
    encoders.load_features(x.to(device))
    break
  return list(encoders.layer_feats.keys())


def collect_single_layer_feats_and_labels(encoders, data_loader, device, n_batches, layer_name):
  layer_feats_list = []
  labels_list = []
  for idx, batch in enumerate(data_loader):
    if idx >= n_batches:
      break

    x, y = batch
    with pt.no_grad():
      encoders.load_features(x.to(device))
    labels_list.append(y)
    layer_feats_list.append(encoders.layer_feats[layer_name].cpu())

  labels = pt.cat(labels_list)
  layer_feats_cat = pt.cat(layer_feats_list)

  return layer_feats_cat, labels


def get_channel_subsets_hsic(encoders: Encoders, channel_filter_rate, data_loader, device,
                             n_batches, n_samples_max, hsic_block_size, hsic_reps):
  layer_feats_dict, labels = collect_feats_and_labels(encoders, data_loader, device, n_batches)

  for layer_name, l_feats in layer_feats_dict.items():
    print("l_feats", l_feats.shape)
    channels_list = hsic_channel_selection(l_feats, labels, channel_filter_rate, n_samples_max,
                                           hsic_block_size, hsic_reps)
    LOG.warning(f'channels_list: {channels_list}')
    encoders.channel_ids[layer_name] = channels_list
    n_layer_feats = math.prod(encoders.layer_feats[layer_name][:, channels_list, :, :].shape[1:])
    encoders.n_feats_by_layer[layer_name] = n_layer_feats


def get_channel_subsets_hsic_low_memory(encoders: Encoders, channel_filter_rate, data_loader,
                                        device, n_batches, n_samples_max, hsic_block_size,
                                        hsic_reps):
  layer_names = get_layer_names(encoders, data_loader, device)

  for layer_name in layer_names:
    l_feats, labels = collect_single_layer_feats_and_labels(encoders, data_loader, device,
                                                            n_batches, layer_name)
    n_feats_per_channel = math.prod(encoders.layer_feats[layer_name].shape[2:])
    encoders.flush_features()
    channels_list = hsic_channel_selection(l_feats, labels, channel_filter_rate, n_samples_max,
                                           hsic_block_size, hsic_reps)
    del l_feats
    LOG.warning(f'channels_list: {channels_list}')
    encoders.channel_ids[layer_name] = channels_list
    n_layer_feats = len(channels_list) * n_feats_per_channel
    encoders.n_feats_by_layer[layer_name] = n_layer_feats


def get_weight_subsets_hsic(encoders: Encoders, channel_filter_rate, data_loader, device,
                            n_batches, hsic_block_size, hsic_reps):
  layer_feats_dict, labels = collect_feats_and_labels(encoders, data_loader, device, n_batches)

  for layer_name, l_feats in layer_feats_dict.items():
    print("l_feats", l_feats.shape)
    # channels_list = hsic_selection(l_feats, labels, channel_filter_rate, n_samples_max_batch)
    weights_list = hsic_weight_selection(l_feats, labels, channel_filter_rate, hsic_block_size,
                                         hsic_reps)
    LOG.warning(f'weights_list: {weights_list}')
    encoders.weight_ids[layer_name] = weights_list
    n_layer_feats = len(weights_list)
    encoders.n_feats_by_layer[layer_name] = n_layer_feats


def get_weight_subsets_hsic_low_memory(encoders: Encoders, channel_filter_rate, data_loader, device,
                                       n_batches, hsic_block_size, hsic_reps):
  layer_names = get_layer_names(encoders, data_loader, device)
  # layer_feats_dict, labels = collect_feats_and_labels(encoders, data_loader, device, n_batches)

  for layer_name in layer_names:
    l_feats, labels = collect_single_layer_feats_and_labels(encoders, data_loader, device,
                                                            n_batches, layer_name)
    encoders.flush_features()
    weights_list = hsic_weight_selection(l_feats, labels, channel_filter_rate, hsic_block_size,
                                         hsic_reps)
    del l_feats
    LOG.warning(f'weights_list: {weights_list}')
    encoders.weight_ids[layer_name] = weights_list
    n_layer_feats = len(weights_list)
    encoders.n_feats_by_layer[layer_name] = n_layer_feats


def hsic_channel_selection(layer: pt.Tensor, labels: pt.Tensor, frac_to_keep: float,
                           n_samples_max: int, hsic_block_size: int, hsic_reps: int):
  """
  @param layer: shape (batch size, C, H, W)
  @param labels: shape (batch size,)
  @param frac_to_keep: in range (0, 1]
  @param n_samples_max: in order to restrict memory consumption, limit n of samples
  """
  bs, n_channels, height, width = layer.shape
  layer = pt.permute(layer, (0, 2, 3, 1)).reshape(-1, n_channels)
  labels = pt.repeat_interleave(labels, height * width, dim=0)

  # subsample if more features than desired
  n_effective_samples = bs * height * width
  if (n_samples_max > 0) and (n_effective_samples > n_samples_max):
    selection = pt.randperm(n_effective_samples)[:n_samples_max]
    layer = layer[selection]
    labels = labels[selection]

  layer = layer.cpu().numpy()
  labels = labels.cpu().numpy()

  channel_order, channel_scores = hsic_lasso(layer, labels, hsic_block_size, hsic_reps)
  n_to_keep = int(n_channels // (1 / frac_to_keep))
  # LOG.warning(f'labels: {labels}')
  # LOG.warning(f'order {channel_order}, n keep {n_to_keep}')
  # LOG.warning(f'channel_scores: {channel_scores}')
  assert len(channel_order) >= n_to_keep
  channels_to_keep = channel_order[:n_to_keep]
  channels_sorted = sorted(channels_to_keep)
  return channels_sorted


def hsic_weight_selection(layer: pt.Tensor, labels: pt.Tensor, frac_to_keep: float,
                          hsic_block_size: int, hsic_reps: int):
  """
    @param layer: shape (batch size, C, H, W)
    @param labels: shape (batch size,)
    @param frac_to_keep: in range (0, 1]
    """
  bs, n_channels, height, width = layer.shape
  layer = layer.reshape(bs, n_channels * height * width)

  layer = layer.cpu().numpy()
  labels = labels.cpu().numpy()

  channel_order, channel_scores = hsic_lasso(layer, labels, hsic_block_size, hsic_reps)
  n_to_keep = int(n_channels // (1 / frac_to_keep))
  # LOG.warning(f'labels: {labels}')
  # LOG.warning(f'order {channel_order}, n keep {n_to_keep}')
  # LOG.warning(f'channel_scores: {channel_scores}')
  assert len(channel_order) >= n_to_keep
  channels_to_keep = channel_order[:n_to_keep]
  channels_sorted = sorted(channels_to_keep)
  return channels_sorted


def hsic_lasso(x: np.ndarray, y: np.ndarray, block_size: int = 100, n_repetitions: int = 3):
  """
  @param x: feature array of shape n_samples x n_features
  @param y: label array of shape n_samples
  @param block_size: size ob minibatch for hsic
  @param n_repetitions: number of epochs for hsic
  """
  n_samples, n_feats = x.shape
  LOG.warning(f'hsic_debug: nsamp {n_samples}, nfeat {n_feats}, block {block_size}, rep {n_repetitions}')
  n_feats_regression = n_feats  # don't fully understand what this does yet
  assert n_samples == len(y)
  hsic_lasso = hsic.HSICLasso()
  hsic_lasso.input(x, y)
  hsic_lasso.classification(n_feats_regression, B=block_size, M=n_repetitions)
  feature_order = hsic_lasso.get_index()
  feature_scores = hsic_lasso.get_index_score()
  return feature_order, feature_scores


def hsic_test():
  hsic_lasso = hsic.HSICLasso()
  n_feats = 11
  n_samples = 40
  x_data = np.random.randn(n_samples, n_feats)
  y_data = np.random.randn(n_samples)
  # print(x_data, y_data)
  featname = [f'Feat{x}' for x in range(n_feats)]
  hsic_lasso.input(x_data, y_data, featname=featname)
  hsic_lasso.regression(n_feats, B=0, M=1)
  hsic_lasso.dump()
  print(hsic_lasso.get_index())
  max_val = hsic_lasso.get_index_score()[0]
  print([k/max_val for k in hsic_lasso.get_index_score()])
  # hsic_lasso.plot_path()
  # hsic_lasso.save_param()

# def main():
#   # Numpy array input example
#   hsic_lasso = hsic.HSICLasso()
#   data = sio.loadmat("../tests/test_data/matlab_data.mat")
#   X = data['X'].transpose()
#   Y = data['Y'][0]
#   featname = ['Feat%d' % x for x in range(1, X.shape[1] + 1)]
#
#   # Get rid of the effect of feature 100 and 300
#   covars_index = np.array([99, 299])
#
#   hsic_lasso.input(X, Y, featname=featname)
#   hsic_lasso.regression(5, covars=X[:, covars_index], covars_kernel="Gaussian")
#   hsic_lasso.dump()
#   hsic_lasso.plot_path()
#
#   # Save parameters
#   hsic_lasso.save_param()


if __name__ == "__main__":
  hsic_test()
