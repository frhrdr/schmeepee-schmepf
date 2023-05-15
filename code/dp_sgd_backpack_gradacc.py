import math
import torch as pt
from backpack import backpack
from backpack.extensions import BatchGrad, BatchL2Grad


def clip_grad_acc(params, large_batch, large_targets, loss_fun, small_batch_size, clip_norm):
  """

  :param params:
  :param large_batch: data batch to be split for gradient accumulation
  :param large_targets: target batch to be split for gradient accumulation, may be none
  :param loss_fun: function that takes a batch (and targets) and returns a loss
  :param small_batch_size: divide large_batch into small batches of this size
  :param clip_norm:
  :return:
  """
  if not isinstance(params, list):
    params = [p for p in params]

  loss_sum = 0
  acc_grad_clipped = None
  global_norms, global_clips = [], []
  n_splits = math.ceil(large_batch.shape[0] / small_batch_size)

  for split_id in range(n_splits):  # get clipped grads for each small batch and accumulate them
    a, b = split_id * small_batch_size, (split_id + 1) * small_batch_size
    split_batch = large_batch[a:b]
    if large_targets is None:
      split_loss = loss_fun(split_batch)
    else:
      split_targets = large_targets[a:b]
      split_loss = loss_fun(split_batch, split_targets)

    loss_sum += split_loss
    split_grad_clipped, split_g_norms, split_g_clips = get_clipped_grad(params, split_loss,
                                                                        clip_norm)
    if acc_grad_clipped is None:
      acc_grad_clipped = split_grad_clipped
    else:
      acc_grad_clipped = [k + j for k, j in zip(acc_grad_clipped, split_grad_clipped)]
    global_norms.append(split_g_norms)
    global_clips.append(split_g_clips)

  # pert_and_apply_grad(params, acc_grad_clipped, noise_factor, clip_norm, device)
  return acc_grad_clipped, loss_sum, pt.cat(global_norms), pt.cat(global_clips)


def pert_and_apply_grad(params, grad_list, noise_factor, clip_norm, device, replace_grad=True):
  noise_sdev = noise_factor * 2 * clip_norm
  for param, c_grad in zip(params, grad_list):
    perturbed_grad = c_grad + pt.randn_like(c_grad, device=device) * noise_sdev  # ...and applied
    if replace_grad:
      param.grad = perturbed_grad  # now we set the parameter gradient to what we just computed
    else:
      param.grad += perturbed_grad


def dp_sgd_backward_gradacc(params, large_batch, large_targets, loss_fun, small_batch_size, device, clip_norm,
                            noise_factor):
  grad, _, norms, clips = clip_grad_acc(params, large_batch, large_targets, loss_fun, small_batch_size, clip_norm)
  pert_and_apply_grad(params, grad, noise_factor, clip_norm, device)
  return norms, clips


def get_clipped_grad(params, loss, clip_norm):
  with backpack(BatchGrad(), BatchL2Grad()):
    loss.backward()

  squared_param_norms = [p.batch_l2 for p in params]  # first we get all the squared parameter norms...
  global_norms = pt.sqrt(pt.sum(pt.stack(squared_param_norms), dim=0))  # ...then compute the global norms...
  global_clips = pt.clamp_max(clip_norm / global_norms, 1.)  # ...and finally get a vector of clipping factors

  clipped_grads = []
  for idx, param in enumerate(params):
    clipped_sample_grads = param.grad_batch * expand_vector(global_clips, param.grad_batch)
    clipped_grad = pt.sum(clipped_sample_grads, dim=0)  # after clipping we sum over the batch
    clipped_grads.append(clipped_grad)
    # noise_sdev = noise_factor * 2 * clip_norm  # gaussian noise standard dev is computed (sensitivity is 2*clip)...
    # perturbed_grad = clipped_grad + pt.randn_like(clipped_grad, device=device) * noise_sdev  # ...and applied
    # param.grad = perturbed_grad  # now we set the parameter gradient to what we just computed

  return clipped_grads, global_norms, global_clips

def expand_vector(vec, tgt_tensor):
  tgt_shape = [vec.shape[0]] + [1] * (len(tgt_tensor.shape) - 1)
  return vec.view(*tgt_shape)
