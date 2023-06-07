import os
import random
import torch as pt
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import argparse
from data_loading import load_imagenet_numpy, load_dataset, IMAGENET_MEAN, IMAGENET_SDEV
from eval_fid import get_fid_scores_fixed
from resnets_groupnorm import ResnetGroupnormG, ResNetFlat
from dp_analysis import find_dpsgd_sigma
# from dp_sgd_backpack import dp_sgd_backward
from backpack import extend
from dp_sgd_backpack_gradacc import clip_grad_acc, pert_and_apply_grad


REAL_LABEL = 1.
FAKE_LABEL = 0.


def get_args():
  parser = argparse.ArgumentParser()

  # PARAMS YOU LIKELY WANT TO SET
  parser.add_argument('--seed', type=int, default=999)
  parser.add_argument('--data', type=str, default='cifar10',
                      choices=['cifar10', 'celeba', 'imagenet32', 'celeba64', 'imagenet64'])
  parser.add_argument('--workers', type=int, default=1)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--batch_size_grad_acc', type=int, default=128)
  parser.add_argument('--nc', type=int, default=3)
  parser.add_argument('--nz', type=int, default=100)
  parser.add_argument('--ngf', type=int, default=64)
  parser.add_argument('--ndf', type=int, default=64)

  parser.add_argument('--n_epochs', type=int, default=20)
  parser.add_argument('--n_save_epochs', type=int, default=5)
  parser.add_argument('--lr_default', type=float, default=0.0002)
  parser.add_argument('--lr_dis', type=float, default=None)
  parser.add_argument('--lr_gen', type=float, default=None)
  parser.add_argument('--beta1', type=float, default=0.5)
  parser.add_argument('--n_gpu', type=int, default=1)
  parser.add_argument('--train_gen_every_n_steps', '-gen_freq', type=int, default=1)

  parser.add_argument('--target_eps', type=float, default=None)
  parser.add_argument('--target_delta', type=float, default=1e-6)
  parser.add_argument('--clip_norm', type=float, default=None)

  parser.add_argument('--model', type=str, default='convnet', choices=['convnet', 'resnet'])
  parser.add_argument('--exp_name', type=str)
  parser.add_argument('--pretrain_checkpoint', type=str, default=None)
  parser.add_argument('--single_iter', action='store_true')
  parser.add_argument('--local_data', action='store_true')

  parser.add_argument('--gen_output', type=str, default='tanh', choices=['linear', 'tanh'])

  arg = parser.parse_args()
  return arg


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_resnets(nz, ngf, ndf, device, img_hw, gen_output):
  # net_d = get_resnet9_discriminator(ndf, affine_gn=False, use_sigmoid=False).to(device)
  net_d = ResNetFlat(planes=ndf, use_sigmoid=False,
                     image_size=img_hw).to(device)
  net_g = ResnetGroupnormG(nz, 3, ngf, img_hw, adapt_filter_size=True,
                           use_conv_at_skip_conn=False, gen_output=gen_output,
                           affine_gn=False).to(device)
  return net_d, net_g


def get_convnets(arg, device):
  net_d = Discriminator(arg.n_gpu, arg.nc, arg.ndf).to(device)
  net_d.apply(weights_init)

  net_g = Generator(arg.n_gpu, arg.nc, arg.nz, arg.ngf).to(device)
  net_g.apply(weights_init)

  return net_d, net_g


# Generator Code
class Generator(nn.Module):
  def __init__(self, ngpu, nc, nz, ngf):
      super(Generator, self).__init__()
      self.ngpu = ngpu
      self.main = nn.Sequential(
          # input is Z, going into a convolution
          nn.ConvTranspose2d(nz, ngf * 8, 3, 1, 0, bias=False),
          nn.GroupNorm(ngf * 8, ngf * 8, affine=False),
          # nn.BatchNorm2d(ngf * 8),
          nn.ReLU(True),
          # state size. (ngf*8) x 4 x 4
          nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 1, 1, bias=False),
          nn.GroupNorm(ngf * 4, ngf * 4, affine=False),
          # nn.BatchNorm2d(ngf * 4),
          nn.ReLU(True),
          # state size. (ngf*4) x 8 x 8
          nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
          nn.GroupNorm(ngf * 2, ngf * 2, affine=False),
          # nn.BatchNorm2d(ngf * 2),
          nn.ReLU(True),
          # state size. (ngf*2) x 16 x 16
          nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
          nn.GroupNorm(ngf, ngf, affine=False),
          # nn.BatchNorm2d(ngf),
          nn.ReLU(True),
          # state size. (ngf) x 32 x 32
          nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
          nn.Tanh()
          # state size. (nc) x 64 x 64
      )

  def forward(self, x):
    return self.main(x)


class Discriminator(nn.Module):
  def __init__(self, ngpu, nc, ndf):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
      # input is (nc) x 32 x 32
      nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=False),
      # state size. (ndf) x 16 x 16
      nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
      nn.GroupNorm(ndf * 2, ndf * 2, affine=False),
      nn.LeakyReLU(0.2, inplace=False),
      # state size. (ndf*2) x 8 x 8
      nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
      nn.GroupNorm(ndf * 4, ndf * 4, affine=False),
      nn.LeakyReLU(0.2, inplace=False),
      # state size. (ndf*4) x 4 x 4
      nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False),
      nn.GroupNorm(ndf * 8, ndf * 8, affine=False),
      nn.LeakyReLU(0.2, inplace=False),
      # state size. (ndf*8) x 3 x 3
      nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False),
      # nn.Sigmoid()
    )

  def forward(self, x):
    return self.main(x)


def expand_vector(vec, tgt_tensor):
  tgt_shape = [vec.shape[0]] + [1] * (len(tgt_tensor.shape) - 1)
  return vec.view(*tgt_shape)


def make_final_plots(dataloader, g_losses, d_losses, device, final_images, save_dir):
  plt.figure(figsize=(10, 5))
  plt.title("Generator and Discriminator Loss During Training")
  plt.plot(g_losses, label="G")
  plt.plot(d_losses, label="D")
  plt.xlabel("iterations")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig(os.path.join(save_dir, 'dcgan_training_loss.png'))

  real_batch = next(iter(dataloader))

  # Plot the real images
  plt.figure(figsize=(15, 15))
  plt.subplot(1, 2, 1)
  plt.axis("off")
  plt.title("Real Images")
  plt.imshow(
    np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),
                 (1, 2, 0)))

  # Plot the fake images from the last epoch
  plt.subplot(1, 2, 2)
  plt.axis("off")
  plt.title("Fake Images")
  plt.imshow(np.transpose(final_images, (1, 2, 0)))
  plt.savefig(os.path.join(save_dir, 'dcgan_real_v_fake.png'))


def create_fid_dataset(net_gen, nz, device, save_dir='.', file_name='synth_data'):
  n_samples = 50_000
  batch_size = 100

  batches = [batch_size] * (n_samples // batch_size)

  samples_list = []
  n_prints = 3
  with pt.no_grad():
    for batch in batches:
      z_in = pt.randn(batch, nz, 1, 1, device=device)
      syn_batch = net_gen(z_in)
      samples_list.append(syn_batch.detach().cpu())
      if n_prints > 0:
        print(f'dpgan dataset ranges: max={pt.max(syn_batch)}, min={pt.min(syn_batch)}')
        n_prints -= 1
    samples = pt.cat(samples_list, dim=0)

  file_name = file_name if file_name.endswith('.npz') else file_name + '.npz'
  file_path = os.path.join(save_dir, file_name)
  np.savez(file_path, x=samples.numpy())
  return file_path


def nondp_update(net_d, net_g, optimizer_d, optimizer_g, real_batch, criterion, label,
                 arg, b_size, device, global_step):
  real_pred = net_d(real_batch).view(-1)
  err_d_real = criterion(real_pred, label)
  err_d_real.backward()
  real_pred_acc = pt.sigmoid(real_pred).mean().item()

  noise = get_gen_noise(b_size, arg.nz, device)
  fake = net_g(noise)
  label.fill_(FAKE_LABEL)
  fake_pred = net_d(fake.detach()).view(-1)
  err_d_fake = criterion(fake_pred, label)
  err_d_fake.backward()
  fake_pred_acc = pt.sigmoid(fake_pred).mean().item()

  err_d = err_d_real + err_d_fake
  err_d_item = err_d.item()
  optimizer_d.step()

  ############################
  # (2) Update G network: maximize log(D(G(z)))
  ###########################
  if global_step % arg.train_gen_every_n_steps == 0:
    net_g.zero_grad()
    label.fill_(REAL_LABEL)  # fake labels are real for generator cost
    fake_pred = net_d(fake).view(-1)
    err_g = criterion(fake_pred, label)
    err_g.backward()
    optimizer_g.step()
  else:
    err_g = None
  return real_pred_acc, fake_pred_acc, err_d_item, err_g


def dp_update(net_d, net_g, optimizer_d, optimizer_g, real_batch, criterion, disc_loss,
              label, arg, b_size, device, global_step, noise_factor):
  grad_d_real, err_d_real, _, _ = clip_grad_acc(net_d.parameters(), real_batch, label,
                                                disc_loss, arg.batch_size_grad_acc,
                                                arg.clip_norm)
  fake = net_g(get_gen_noise(b_size, arg.nz, device))
  label.fill_(FAKE_LABEL)
  grad_d_fake, err_d_fake, _, _ = clip_grad_acc(net_d.parameters(), fake.detach(), label,
                                                disc_loss, arg.batch_size_grad_acc,
                                                arg.clip_norm)
  pert_and_apply_grad(net_d.parameters(), grad_d_real, noise_factor, arg.clip_norm, device,
                      replace_grad=True)
  pert_and_apply_grad(net_d.parameters(), grad_d_fake, 0., arg.clip_norm, device,
                      replace_grad=False)

  optimizer_d.step()

  ############################
  # (2) Update G network: maximize log(D(G(z)))
  ###########################
  if global_step % arg.train_gen_every_n_steps == 0:
    net_d.zero_grad()
    net_g.zero_grad()
    label.fill_(REAL_LABEL)  # fake labels are real for generator cost
    # fake = net_g(pt.randn(b_size, arg.nz, 1, 1, device=device))
    fake_pred = net_d(fake).view(-1)
    err_g = criterion(fake_pred, label)
    err_g.backward()
    optimizer_g.step()


def get_gen_noise(b_size, nz, device):
  return pt.randn(b_size, nz, 1, 1, device=device)


def create_checkpoint(log_dir, epoch, noise_scale, best_fid, best_epoch, best_syn_data_path,
                      opt_g, opt_d, net_g, net_d, global_step):
  ckpt = dict()
  ckpt['epoch'] = epoch
  ckpt['noise_scale'] = noise_scale
  ckpt['best_fid'] = best_fid
  ckpt['best_epoch'] = best_epoch
  ckpt['best_syn_data_path'] = best_syn_data_path
  ckpt['global_step'] = global_step
  ckpt['opt_g'] = opt_g.state_dict()
  ckpt['opt_d'] = opt_d.state_dict()
  ckpt['net_g'] = net_g.state_dict()
  ckpt['net_d'] = net_d.state_dict()

  pt.save(ckpt, os.path.join(log_dir, f'checkpoint.pt'))
  pt.save(ckpt, os.path.join(log_dir, f'checkpoint_ep{epoch}.pt'))


def load_checkpoint(checkpoint_path):
  ckpt = pt.load(checkpoint_path)
  start_epoch = ckpt['epoch'] + 1
  noise_scale = ckpt['noise_scale']
  best_fid = ckpt['best_fid']
  best_epoch = ckpt['best_epoch']
  best_syn_data_path = ckpt['best_syn_data_path']
  opt_g_state = ckpt['opt_g']
  opt_d_state = ckpt['opt_d']
  net_g_state = ckpt['net_g']
  net_d_state = ckpt['net_d']
  global_step = ckpt['global_step']
  return start_epoch, noise_scale, best_fid, best_epoch, best_syn_data_path, \
      opt_g_state, opt_d_state, net_g_state, net_d_state, global_step


def print_fake_batch(arg, fake, data_scale, save_dir, file_name='clamped_plot.png'):
  if arg.data in {'cifar10', 'celeba'}:
    if data_scale == 'normed':
      mean_tsr = pt.tensor(IMAGENET_MEAN, device=fake.device)
      sdev_tsr = pt.tensor(IMAGENET_SDEV, device=fake.device)
      data_to_print = fake * sdev_tsr[None, :, None, None] + mean_tsr[None, :, None, None]
    else:
      data_to_print = fake
  else:
    mean_tsr = pt.tensor([0.5, 0.5, 0.5], device=fake.device)
    sdev_tsr = pt.tensor([0.5, 0.5, 0.5], device=fake.device)
    data_to_print = fake * sdev_tsr[None, :, None, None] + mean_tsr[None, :, None, None]

  print(f'data value range prior to clamping: min={pt.min(data_to_print)}, max={pt.max(data_to_print)}')
  data_to_print = pt.clamp(data_to_print, min=0., max=1.)
  img_path_clamp = os.path.join(save_dir, file_name)
  vutils.save_image(data_to_print, img_path_clamp, normalize=True, nrow=10)


def update_best_score(fid, syn_data_path, epoch, best_fid, best_syn_data_path, best_epoch):
  if best_fid is None or fid < best_fid:  # new best score
    if best_syn_data_path is not None:
      try:
        os.remove(best_syn_data_path)
      except FileNotFoundError:
        print(f'failed to delete syn data {best_syn_data_path}')
    best_fid = fid
    best_epoch = epoch
    best_syn_data_path = syn_data_path
  else:  # best score in the past
    try:
      os.remove(syn_data_path)
    except FileNotFoundError:
      print(f'failed to delete syn data {syn_data_path}')
  return best_fid, best_syn_data_path, best_epoch


def main():
  arg = get_args()
  save_dir = os.path.join('../logs/dcgan/', arg.exp_name)

  os.makedirs(save_dir, exist_ok=True)
  print("Random Seed: ", arg.seed)
  random.seed(arg.seed)
  pt.manual_seed(arg.seed)

  checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
  if os.path.exists(checkpoint_path):
    start_epoch, noise_factor, best_fid, best_epoch, best_syn_data_path, opt_g_state, opt_d_state, \
      net_g_state, net_d_state, global_step = load_checkpoint(checkpoint_path)
  else:
    start_epoch = 0
    global_step = 0
    noise_factor = None
    best_fid = None
    best_epoch = None
    best_syn_data_path = None
    opt_g_state = None
    opt_d_state = None
    net_g_state = None
    net_d_state = None

  if arg.target_eps is not None and noise_factor is None:
    n_samples_dict = {'cifar10': 50_000, 'imagenet32': 1_281_167, 'celeba': 202_599,
                      'imagenet64': 1_281_167, 'celeba64': 202_599}
    n_samples = n_samples_dict[arg.data]
    n_iter = np.ceil(n_samples / arg.batch_size) * arg.n_epochs
    actual_eps, noise_factor = find_dpsgd_sigma(arg.target_eps, arg.target_delta, arg.batch_size,
                                                n_samples, n_iter)
    print(f'using sigma={noise_factor} to achieve eps={actual_eps}')

  data_root = '/tmp' if arg.local_data else '../data'
  img_hw = 32
  data_scale = 'normed' if arg.gen_output == 'linear' else '0_1'
  if arg.data == 'cifar10':
    # dataloader, _ = load_cifar10(32, data_root, arg.batch_size, arg.workers, 'normed', True, False)
    dataloader, _ = load_dataset('cifar10', img_hw, img_hw, data_root, arg.batch_size, arg.workers,
                                 data_scale, False, False)
  elif arg.data == 'celeba':
    dataloader, _ = load_dataset('celeba', img_hw, img_hw, data_root, arg.batch_size, arg.workers,
                                 data_scale, False, False)
  elif arg.data == 'celeba64':
    img_hw = 64
    dataloader, _ = load_dataset('celeba', img_hw, img_hw, data_root, arg.batch_size, arg.workers,
                                 data_scale, False, False)
    arg.data = 'celeba'
  elif arg.data == 'imagenet64':
    img_hw = 64
    dataloader = load_imagenet_numpy(data_root, arg.batch_size, arg.workers, img_hw=img_hw)
  elif arg.data == 'imagenet32':
    dataloader = load_imagenet_numpy(data_root, arg.batch_size, arg.workers, img_hw=32)
  else:
    raise ValueError

  device = pt.device("cuda:0" if (pt.cuda.is_available() and arg.n_gpu > 0) else "cpu")

  if arg.model == 'resnet':
    net_d, net_g = get_resnets(arg.nz, arg.ngf, arg.ndf, device, img_hw, arg.gen_output)
  else:
    assert img_hw == 32
    net_d, net_g = get_convnets(arg, device)
  if net_g_state is not None and net_d_state is not None:
    net_g.load_state_dict(net_g_state)
    net_d.load_state_dict(net_d_state)
    del net_g_state
    del net_d_state
    print('loaded model weights from checkpoint')
  elif arg.pretrain_checkpoint is not None:
    weights_file = os.path.join('../logs/dcgan/', arg.pretrain_checkpoint)
    pretrained_weights = pt.load(weights_file)
    net_g.load_state_dict(pretrained_weights['net_g'])
    net_d.load_state_dict(pretrained_weights['net_d'])
    del pretrained_weights
    print('loaded model weights from pre-trained model')

  criterion = nn.BCEWithLogitsLoss()

  def disc_loss(x, y):
    return criterion(net_d(x).view(-1), y)

  lr_dis = arg.lr_default if arg.lr_dis is None else arg.lr_dis
  lr_gen = arg.lr_default if arg.lr_gen is None else arg.lr_gen

  if arg.clip_norm is not None:
    net_d = extend(net_d)
    criterion = extend(criterion)
    # net_g = extend(net_g, use_converter=True)

  opt_d = optim.Adam(net_d.parameters(), lr=lr_dis, betas=(arg.beta1, 0.999))
  opt_g = optim.Adam(net_g.parameters(), lr=lr_gen, betas=(arg.beta1, 0.999))

  if opt_g_state is not None and opt_d_state is not None:
    opt_g.load_state_dict(opt_g_state)
    opt_d.load_state_dict(opt_d_state)
    del opt_g_state
    del opt_d_state
    print('loaded optimizer weights from checkpoint')
  # Training Loop

  # Lists to keep track of progress
  # img_list = []
  g_losses = []
  d_losses = []

  print("Starting Training Loop...")

  err_g = None
  real_pred_acc, fake_pred_acc = None, None
  for epoch in range(start_epoch, arg.n_epochs):
    for i, data in enumerate(dataloader, 0):
      global_step += 1
      ############################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ###########################
      # Train with all-real batch

      real_batch = data[0].to(device)
      b_size = real_batch.size(0)
      net_d.zero_grad()
      disc_label = pt.full((b_size,), REAL_LABEL, dtype=pt.float, device=device)

      if arg.clip_norm is None:
        res = nondp_update(net_d, net_g, opt_d, opt_g, real_batch, criterion,
                           disc_label, arg, b_size, device,
                           global_step)
        real_pred_acc, fake_pred_acc, err_d_item, err_g = res
      else:  # PRIVATE SETTING
        dp_update(net_d, net_g, opt_d, opt_g, real_batch, criterion,
                  disc_loss, disc_label, arg, b_size, device,
                  global_step, noise_factor)
        err_d_item = 0

      err_g_item = 0 if err_g is None else err_g.item()
      # Output training stats
      if i % 100 == 0:
        if real_pred_acc is not None:
          print(f'real_pred_acc {real_pred_acc}, fake pred acc {fake_pred_acc}')
        else:
          print(f'err_d {err_d_item}, err_g {err_g_item}')

      # Save Losses for plotting later
      g_losses.append(err_g_item)
      d_losses.append(err_d_item)

      if arg.single_iter:
        break

    if arg.single_iter or (epoch + 1) % arg.n_save_epochs == 0 or epoch + 1 == arg.n_epochs:
      create_checkpoint(save_dir, epoch, noise_factor, best_fid, best_epoch, best_syn_data_path,
                        opt_g, opt_d, net_g, net_d, global_step)
      # make an image
      fake = net_g(get_gen_noise(100, arg.nz, device)).detach().cpu()
      print_fake_batch(arg, fake, data_scale, save_dir, file_name=f'clamped_plot_ep{epoch}.png')
      # create a dataset
      syn_data_path = create_fid_dataset(net_g, arg.nz, device, save_dir, file_name=f'synth_data_ep{epoch}')
      # perform an fid eval
      fid = get_fid_scores_fixed(syn_data_path, dataset_name=arg.data, device=device, n_samples=50_000,
                                 image_size=img_hw, center_crop_size=img_hw, data_scale=data_scale, batch_size=100)
      print(f'fid={fid} at epoch={epoch}')
      np.save(os.path.join(save_dir, f'fid_ep{epoch}.npy'), fid)

      best_fid, best_syn_data_path, best_epoch = update_best_score(fid, syn_data_path, epoch, best_fid,
                                                                   best_syn_data_path, best_epoch)

    if (epoch + 1) % arg.n_save_epochs == 0:
      # sign off with error code 3
      print(f'preparing restart after epoch {epoch}')
      exit(3)

    if arg.single_iter:
      exit(0)


if __name__ == '__main__':
  main()
