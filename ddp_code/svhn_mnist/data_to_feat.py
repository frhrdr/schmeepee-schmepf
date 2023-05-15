import torch
import numpy as np

def data_to_feat(data_batch, net_enc, n_matching_layers, which_layer_to_use, selected_idx=None):
                     # dp_sens_bound, dp_sens_bound_type,

  """
  Applies feature extractor. Concatenate feature vectors from all selected layers.
  """
  # gets features from each layer of net_enc
  features = []

  out_data, feats_per_layer = net_enc(data_batch)

  count = 0
  for layer_id in which_layer_to_use:
    corrected_layer_id = layer_id - 1  # gets features in forward order
    # print('corrected_layer_id', corrected_layer_id)
    layer_feats = feats_per_layer[corrected_layer_id]

    if selected_idx is not None:
      selected_idcs = selected_idx[count]
      layer_feats = layer_feats[:, selected_idcs, :, :]
      count = count + 1

    layer_feats = layer_feats.view(layer_feats.shape[0], -1)

    if n_matching_layers==1:
      features = layer_feats
    else:
      if corrected_layer_id == 0:
        features = layer_feats
      else:
        features = torch.hstack((features, layer_feats))

  return features
