import torch
import numpy as np

def data_to_feat_compress(data_batch, net_enc, n_matching_layers, which_layer_to_use, compression_rate):
                     # dp_sens_bound, dp_sens_bound_type,

  """
  Applies feature extractor. Concatenate feature vectors from all selected layers.
  """
  # gets features from each layer of net_enc
  features = []
  selected_ids_all = []

  out_data, feats_per_layer = net_enc(data_batch)

  for layer_id in which_layer_to_use:
    corrected_layer_id = layer_id - 1  # gets features in forward order
    # print('corrected_layer_id', corrected_layer_id)
    layer_feats = feats_per_layer[corrected_layer_id]

    layer_feats_per_channel = layer_feats.view(layer_feats.shape[0], layer_feats.shape[1], -1)
    avg_norm_per_channel = torch.mean(torch.norm(layer_feats_per_channel, dim=2),dim=0)
    vals, idcs = torch.sort(avg_norm_per_channel, descending=True)
    selected_idcs = idcs[0:int(len(idcs)*compression_rate/100)]
    selected_ids_all.append(selected_idcs)

    selected_feats = layer_feats[:, selected_idcs, :, :]
    layer_feats = selected_feats.view(layer_feats.shape[0], -1)

    if n_matching_layers==1:
      features = layer_feats
    else:
      if corrected_layer_id == 0:
        features = layer_feats
      else:
        features = torch.hstack((features, layer_feats))

  return features, selected_ids_all
