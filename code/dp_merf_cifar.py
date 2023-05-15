import os
import torch as pt
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
from collections import namedtuple
from models.generators import ResnetG
from data_loading import load_cifar10
from eval_fid import get_fid_scores
from dp_analysis import find_single_release_sigma
from torchvision import utils as vutils
rff_param_tuple = namedtuple('rff_params', ['w', 'b'])


def flat_data(data, labels, device, n_labels=10, add_label=False):
  bs = data.shape[0]
  if add_label:
    gen_one_hots = pt.zeros(bs, n_labels, device=device)
    gen_one_hots.scatter_(1, labels[:, None], 1)
    labels = gen_one_hots
    return pt.cat([pt.reshape(data, (bs, -1)), labels], dim=1)
  else:
    if len(data.shape) > 2:
      return pt.reshape(data, (bs, -1))
    else:
      return data


def rff_sphere(x, rff_params):
  """
  this is a Pytorch version of anon's code for RFFKGauss
  Fourier transform formula from http://mathworld.wolfram.com/FourierTransformGaussian.html
  """
  w = rff_params.w
  xwt = pt.mm(x, w.t())
  z_1 = pt.cos(xwt)
  z_2 = pt.sin(xwt)
  z_cat = pt.cat((z_1, z_2), 1)
  norm_const = pt.sqrt(pt.tensor(w.shape[0]).to(pt.float32))
  z = z_cat / norm_const  # w.shape[0] == n_features / 2
  return z


def weights_sphere(d_rff, d_enc, sig, device):
  w_freq = pt.tensor(np.random.randn(d_rff // 2, d_enc) / np.sqrt(sig)).to(pt.float32).to(device)
  return rff_param_tuple(w=w_freq, b=None)


def rff_rahimi_recht(x, rff_params):
  """
  implementation more faithful to rahimi+recht paper
  """
  w = rff_params.w
  b = rff_params.b
  xwt = pt.mm(x, w.t()) + b
  z = pt.cos(xwt)
  z = z * pt.sqrt(pt.tensor(2. / w.shape[0]).to(pt.float32))
  return z


def weights_rahimi_recht(d_rff, d_enc, sig, device):
  w_freq = pt.tensor(np.random.randn(d_rff, d_enc) / np.sqrt(sig)).to(pt.float32).to(device)
  b_freq = pt.tensor(np.random.rand(d_rff) * (2 * np.pi * sig)).to(device)
  return rff_param_tuple(w=w_freq, b=b_freq)


def data_label_embedding(data, labels, rff_params, mmd_type,
                         labels_to_one_hot=False, n_labels=None, device=None, reduce='mean'):
  assert reduce in {'mean', 'sum'}
  if labels_to_one_hot:
    batch_size = data.shape[0]
    one_hots = pt.zeros(batch_size, n_labels, device=device)
    one_hots.scatter_(1, labels[:, None], 1)
    labels = one_hots

  if mmd_type == 'sphere':
    data_embedding = rff_sphere(data, rff_params)
  else:
    data_embedding = rff_rahimi_recht(data, rff_params)

  embedding = pt.einsum('ki,kj->kij', [data_embedding, labels])
  return pt.mean(embedding, 0) if reduce == 'mean' else pt.sum(embedding, 0)


def get_rff_loss(train_loader, n_features, d_rff, rff_sigma, device, n_labels, noise_factor,
                 mmd_type):
  assert d_rff % 2 == 0
  assert mmd_type in {'sphere', 'r+r'}
  if mmd_type == 'sphere':
    w_freq = weights_sphere(d_rff, n_features, rff_sigma, device)
  else:
    w_freq = weights_rahimi_recht(d_rff, n_features, rff_sigma, device)

  noisy_emb = noisy_dataset_embedding(train_loader, w_freq, d_rff, device, n_labels, noise_factor,
                                      mmd_type)

  def single_release_loss(gen_data, gen_labels):
    gen_data = flat_data(gen_data, gen_labels, device, n_labels=10, add_label=False)
    gen_emb = data_label_embedding(gen_data, gen_labels, w_freq, mmd_type)
    return pt.sum((noisy_emb - gen_emb) ** 2)

  return single_release_loss


def noisy_dataset_embedding(train_loader, w_freq, d_rff, device, n_labels, noise_factor, mmd_type,
                            sum_frequency=25):
  emb_acc = []
  n_data = 0

  for data, labels in train_loader:
    data, labels = data.to(device), labels.to(device)
    data = flat_data(data, labels, device, n_labels=10, add_label=False)

    emb_acc.append(data_label_embedding(data, labels, w_freq, mmd_type, labels_to_one_hot=True,
                                        n_labels=n_labels, device=device, reduce='sum'))
    # emb_acc.append(pt.sum(pt.einsum('ki,kj->kij', [rff_gauss(data, w_freq), one_hots]), 0))
    n_data += data.shape[0]

    if len(emb_acc) > sum_frequency:
      emb_acc = [pt.sum(pt.stack(emb_acc), 0)]

  print('done collecting batches, n_data', n_data)
  emb_acc = pt.sum(pt.stack(emb_acc), 0) / n_data
  print(pt.norm(emb_acc), emb_acc.shape)
  noise = pt.randn(d_rff, n_labels, device=device) * (2 * noise_factor / n_data)
  noisy_emb = emb_acc + noise
  return noisy_emb


def get_generator(z_dim):
  net_gen = ResnetG(z_dim, nc=3, ndf=64, image_size=32,
                    adapt_filter_size=True,
                    use_conv_at_skip_conn=False)
  return net_gen


def train_dpmerf(gen, optimizer, scheduler, rff_mmd_loss, n_iter, batch_size,
                 log_interval, scheduler_interval, code_fun,
                 fixed_noise, exp_name, log_dir, n_classes):
  for step in range(n_iter):
    gen_code, gen_labels = code_fun(batch_size)
    loss = rff_mmd_loss(gen(gen_code), gen_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % log_interval == 0 and fixed_noise is not None:
      log_step(step, loss, gen, fixed_noise, exp_name, log_dir, n_classes)

    if step % scheduler_interval == 0:
      scheduler.step()


def log_step(step, loss, gen, fixed_noise, exp_name, log_dir, n_classes):
  print(f'Train step: {step} \tLoss: {loss.item():.6f}')
  with pt.no_grad():
    fake = gen(fixed_noise).detach()
  img_dir = os.path.join(log_dir, 'images')
  os.makedirs(img_dir, exist_ok=True)
  img_path = os.path.join(img_dir, f"{exp_name.replace('/', '_', 999)}_it_{step + 1}.png")
  n_rows = fake.data.shape[0] // n_classes
  if n_rows < fake.data.shape[0] / n_classes:
    n_rows += 1
  vutils.save_image(fake.data, img_path, normalize=True, nrow=fake.data.shape[0] // n_classes)


def compute_rff_loss(gen, data, labels, rff_mmd_loss, device):
  bs = labels.shape[0]
  gen_code, gen_labels = gen.get_code(bs, device)
  gen_samples = gen(gen_code)
  return rff_mmd_loss(data, labels, gen_samples, gen_labels)


def get_args():
  parser = argparse.ArgumentParser()

  # BASICS
  parser.add_argument('--seed', type=int, default=None, help='sets random seed')
  parser.add_argument('--log-interval', type=int, default=100, help='print updates after n steps')
  parser.add_argument('--base-log-dir', type=str, default='../logs/dpmerf/',
                      help='path where logs for all runs are stored')
  parser.add_argument('--log-name', type=str, default='test', help='subdirectory for this run')
  parser.add_argument('--log-dir', type=str, default=None,
                      help='override save path. constructed if None')

  # OPTIMIZATION
  parser.add_argument('--batch-size', '-bs', type=int, default=100)
  parser.add_argument('--test-batch-size', '-tbs', type=int, default=100)
  parser.add_argument('--gen-batch-size', '-gbs', type=int, default=100)
  parser.add_argument('--n_iter', type=int, default=10_000)
  parser.add_argument('--lr', '-lr', type=float, default=0.01, help='learning rate')
  parser.add_argument('--lr-decay', type=float, default=0.9, help='learning rate decay factor')
  parser.add_argument('--scheduler-interval', type=int, default=1_000,
                      help='reduce lr after n steps')

  # MODEL DEFINITION
  parser.add_argument('--d-code', '-dcode', type=int, default=5, help='random code dimensionality')

  # DP SPEC
  parser.add_argument('--d-rff', type=int, default=1000,
                      help='number of random filters for apprixmate mmd')
  parser.add_argument('--rff-sigma', '-rffsig', type=float, default=100.,
                      help='standard dev. for filter sampling')
  parser.add_argument('--tgt-eps', type=float, default=None, help='privacy parameter - finds noise')

  parser.add_argument('--mmd-type', type=str, default='sphere', help='how to approx mmd',
                      choices=['sphere', 'r+r'])
  ar = parser.parse_args()

  preprocess_args(ar)
  return ar


def preprocess_args(ar):
  if ar.log_dir is None:
    assert ar.log_name is not None
    ar.log_dir = ar.base_log_dir + ar.log_name + '/'

  os.makedirs(ar.log_dir, exist_ok=True)

  if ar.seed is None:
    ar.seed = np.random.randint(0, 1000)


def synthesize_data_with_uniform_labels(gen, device, code_fun, gen_batch_size=1000, n_data=60000,
                                        n_labels=10):
  gen.eval()
  if n_data % gen_batch_size != 0:
    assert n_data % 100 == 0
    gen_batch_size = n_data // 100
  assert gen_batch_size % n_labels == 0
  n_iterations = n_data // gen_batch_size

  data_list = []
  ordered_labels = pt.repeat_interleave(pt.arange(n_labels), gen_batch_size // n_labels)[:, None].to(device)
  labels_list = [ordered_labels] * n_iterations

  with pt.no_grad():
    for idx in range(n_iterations):
      gen_code = code_fun(gen_batch_size, labels=ordered_labels)
      gen_samples = gen(gen_code)
      data_list.append(gen_samples)
  return pt.cat(data_list, dim=0).cpu().numpy(), pt.cat(labels_list, dim=0).cpu().numpy()


def get_code_fun(device, n_labels, d_code):

  def get_code(batch_size, labels=None):
    return_labels = False
    if labels is None:  # sample labels
      return_labels = True
      labels = pt.randint(n_labels, (batch_size, 1), device=device)
    code = pt.randn(batch_size, d_code, device=device)
    gen_one_hots = pt.zeros(batch_size, n_labels, device=device)
    gen_one_hots.scatter_(1, labels, 1)
    code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)[:, :, None, None]
    # print(code.shape)
    if return_labels:
      return code, gen_one_hots
    else:
      return code
  return get_code


def main():
  # load settings
  ar = get_args()
  pt.manual_seed(ar.seed)
  use_cuda = pt.cuda.is_available()
  device = pt.device("cuda" if use_cuda else "cpu")
  if ar.tgt_eps is not None:
    _, noise_factor = find_single_release_sigma(ar.tgt_eps, target_delta=1e-5)
  else:
    noise_factor = 0.
  # load data
  train_loader, n_classes = load_cifar10(image_size=32, dataroot='../data/',
                                         batch_size=ar.batch_size, n_workers=2, data_scale='normed',
                                         labeled=True, test_set=False)
  image_size = 32
  n_features = image_size * image_size * 3

  gen = get_generator(ar.d_code + n_classes)
  gen.to(device)

  rff_mmd_loss = get_rff_loss(train_loader, n_features, ar.d_rff, ar.rff_sigma, device, n_classes,
                              noise_factor, ar.mmd_type)
  # init optimizer
  optimizer = pt.optim.Adam(list(gen.parameters()), lr=ar.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)

  code_fun = get_code_fun(device, n_classes, ar.d_code)

  # for plotting samples - turn off by setting fixed_noise to None
  n_plot_cols = 10
  fixed_labels = pt.repeat_interleave(pt.arange(n_classes, device=device), n_plot_cols)
  fixed_noise = code_fun(batch_size=len(fixed_labels), labels=fixed_labels[:, None])

  train_dpmerf(gen, optimizer, scheduler, rff_mmd_loss, ar.n_iter, ar.batch_size,
               ar.log_interval, ar.scheduler_interval, code_fun,
               fixed_noise, ar.log_name, ar.log_dir, n_classes)

  # save trained model and data
  pt.save(gen.state_dict(), ar.log_dir + 'gen.pt')

  data_file = os.path.join(ar.log_dir, 'gen_data.npz')
  syn_data_size = 5000
  syn_data, syn_labels = synthesize_data_with_uniform_labels(gen, device, code_fun,
                                                             ar.gen_batch_size, syn_data_size,
                                                             n_classes)

  np.savez(data_file, x=syn_data, y=syn_labels)

  fid_score = get_fid_scores(data_file, 'cifar10', device, syn_data_size,
                             image_size, center_crop_size=32, data_scale='normed',
                             base_data_dir='../data', batch_size=50)
  print(f'fid={fid_score}')
  np.save(os.path.join(ar.log_dir, 'fid.npy'), fid_score)


if __name__ == '__main__':
  main()
