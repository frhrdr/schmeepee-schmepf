import torch as pt
from util_logging import LOG
from models.encoders_class import Encoders
from feature_selection_hsic import get_layer_names, collect_single_layer_feats_and_labels


def get_pca_reduction(encoders: Encoders, channel_filter_rate, data_loader, device, n_batches,
                      n_pca_iter):
  layer_names = get_layer_names(encoders, data_loader, device)

  for layer_name in layer_names:
    l_feats, _ = collect_single_layer_feats_and_labels(encoders, data_loader, device, n_batches,
                                                       layer_name)
    l_feats = l_feats.to(device)
    # print("l_feats", l_feats.shape)
    # channels_list = hsic_selection(l_feats, labels, channel_filter_rate, n_samples_max_batch)
    bs, n_channels, height, width = l_feats.shape
    feats_dim = n_channels * height * width
    l_feats = l_feats.reshape(bs, feats_dim)
    approx_dim = int(feats_dim // (1 / channel_filter_rate))
    LOG.debug(f'layer {layer_name} dims: {bs}, {n_channels}, {height}, {width}, '
              f'total {feats_dim}, reduced {approx_dim}')

    # center data and store centering op
    l_feats_mean = pt.mean(l_feats, dim=0, keepdim=True)
    l_feats_sdev = pt.std(l_feats, dim=0, keepdim=True)
    l_feats_centered = pt.where(l_feats_sdev > 0, ((l_feats - l_feats_mean) / l_feats_sdev),
                                pt.zeros_like(l_feats))

    # LOG.warning(f'starting pca for {layer_name}')
    assert not pt.any(pt.isnan(l_feats_centered))
    _, _, projection_v = pt.pca_lowrank(l_feats_centered, q=approx_dim, center=False,
                                        niter=n_pca_iter)
    # LOG.warning(f'done with pca for {layer_name}')

    encoders.pca_maps[layer_name] = PCAMapping(l_feats_mean, l_feats_sdev, projection_v)
    encoders.n_feats_by_layer[layer_name] = approx_dim
    del l_feats
    del l_feats_centered


class PCAMapping:
  def __init__(self, mean: pt.Tensor, sdev: pt.Tensor, map_v: pt.Tensor):
    self.mean = mean
    self.sdev = sdev
    self.map_v = map_v

  def map_batch(self, batch: pt.Tensor) -> pt.Tensor:
    batch = pt.reshape(batch, (batch.shape[0], -1))
    batch_centered = (batch - self.mean) / self.sdev
    return pt.matmul(batch_centered, self.map_v)

  def to_param_tuple(self):
    return self.mean, self.sdev, self.map_v

  @staticmethod
  def from_param_tuple(params):
    mean, sdev, map_v = params
    return PCAMapping(mean, sdev, map_v)
