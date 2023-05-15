import os
import random
import torch as pt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import argparse
from data_loading import load_imagenet_numpy, load_cifar10
from eval_fid import get_fid_scores
from models.resnets_groupnorm import ResnetGroupnormG, ResNetFlat
from dp_analysis import find_dpsgd_sigma
# from dp_sgd_backpack import dp_sgd_backward
from dp_sgd_backpack_gradacc import clip_grad_acc, pert_and_apply_grad
from backpack import extend


REAL_LABEL = 1.
FAKE_LABEL = 0.


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_resnets(nz, ngf, ndf, n_classes, device, img_hw):
  # net_d = get_resnet9_discriminator(ndf, affine_gn=False, use_sigmoid=False).to(device)
  net_d = ResNetFlat(planes=ndf, n_classes=n_classes, use_sigmoid=False,
                     image_size=img_hw).to(device)
  net_g = ResnetGroupnormG(nz, 3, ngf, img_hw, adapt_filter_size=True,
                           use_conv_at_skip_conn=False, gen_output='linear',
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


# def load_cifar10(dataroot, batch_size,
#                  n_workers, test_set=False):
#   transformations = [transforms.ToTensor(),
#                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
#   dataset = dset.CIFAR10(root=dataroot, train=not test_set, download=True,
#                          transform=transforms.Compose(transformations))
#
#   dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                         shuffle=True, num_workers=int(n_workers))
#
#   return dataloader


def load_celeba(dataroot, batch_size, n_workers, image_size=32, center_crop_size=32):

  transformations = []
  if center_crop_size > image_size:
    transformations.extend([transforms.CenterCrop(center_crop_size),
                            transforms.Resize(image_size)])
  else:
    transformations.extend([transforms.Resize(image_size),
                            transforms.CenterCrop(center_crop_size)])

  transformations.extend([transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])])

  # folder dataset
  dataset = dset.ImageFolder(root=os.path.join(dataroot, 'img_align_celeba'),
                             transform=transforms.Compose(transformations))

  dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=int(n_workers))

  return dataloader


def expand_vector(vec, tgt_tensor):
  tgt_shape = [vec.shape[0]] + [1] * (len(tgt_tensor.shape) - 1)
  return vec.view(*tgt_shape)


def make_final_plots(dataloader, G_losses, D_losses, device, final_images, save_dir):
  plt.figure(figsize=(10, 5))
  plt.title("Generator and Discriminator Loss During Training")
  plt.plot(G_losses, label="G")
  plt.plot(D_losses, label="D")
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
  with pt.no_grad():
    for batch in batches:
      z_in = pt.randn(batch, nz, 1, 1, device=device)
      syn_batch = net_gen(z_in)
      samples_list.append(syn_batch.detach().cpu())
    samples = pt.cat(samples_list, dim=0)

  file_name = file_name if file_name.endswith('.npz') else file_name + '.npz'
  file_path = os.path.join(save_dir, file_name)
  np.savez(file_path, x=samples.numpy())
  return file_path


def nondp_unlabeled_update(net_d, net_g, optimizer_d, optimizer_g, real_batch, criterion, label,
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


def dp_unlabeled_update(net_d, net_g, optimizer_d, optimizer_g, real_batch, criterion, disc_loss,
                        label, arg, b_size, device, global_step,
                        noise_factor):
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
  err_d_item = 0
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


def get_gen_noise(b_size, nz, device, n_classes=None, labels=None):
  if n_classes is None:
    return pt.randn(b_size, nz, 1, 1, device=device)
  else:
    gen_noise = pt.randn(b_size, nz - n_classes, 1, 1, device=device)
    if labels is None:
      labels = pt.eye(n_classes, device=device)[pt.randint(n_classes, (b_size,))]
    labeled_noise = pt.cat((labels[:, :, None, None], gen_noise), dim=1)
    return labeled_noise, labels


def nondp_labeled_update(net_d, net_g, optimizer_d, optimizer_g, real_batch, criterion, label,
                         arg, b_size, device, global_step, real_labels, n_classes):
  # print(real_labels.shape)
  real_labels_onehot = pt.eye(n_classes, device=device)[real_labels]
  real_pred = net_d(real_batch, real_labels_onehot).view(-1)
  err_d_real = criterion(real_pred, label)
  err_d_real.backward()
  real_pred_acc = pt.sigmoid(real_pred).mean().item()

  labeled_noise, rand_labels = get_gen_noise(b_size, arg.nz, device, n_classes)
  fake = net_g(labeled_noise)
  label.fill_(FAKE_LABEL)
  fake_pred = net_d(fake.detach(), rand_labels).view(-1)
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
    fake_pred = net_d(fake, rand_labels).view(-1)
    err_g = criterion(fake_pred, label)
    err_g.backward()
    optimizer_g.step()
  else:
    err_g = None
  return real_pred_acc, fake_pred_acc, err_d_item, err_g


def dp_labeled_update(net_d, net_g, optimizer_d, optimizer_g, real_batch, criterion, disc_loss,
                      label, arg, b_size, device, global_step, noise_factor, real_labels,
                      n_classes):
  real_labels_onehot = pt.eye(n_classes, device=device)[real_labels]
  grad_d_real, err_d_real, _, _ = clip_grad_acc(net_d.parameters(), real_batch, real_labels_onehot,
                                                lambda x, t: disc_loss(x, t, label[:x.shape[0]]),
                                                arg.batch_size_grad_acc, arg.clip_norm)
  labeled_noise, rand_labels = get_gen_noise(b_size, arg.nz, device, n_classes)
  fake = net_g(labeled_noise)
  label.fill_(FAKE_LABEL)
  grad_d_fake, err_d_fake, _, _ = clip_grad_acc(net_d.parameters(), fake.detach(), rand_labels,
                                                lambda x, t: disc_loss(x, t, label[:x.shape[0]]),
                                                arg.batch_size_grad_acc, arg.clip_norm)
  pert_and_apply_grad(net_d.parameters(), grad_d_real, noise_factor, arg.clip_norm, device,
                      replace_grad=True)
  pert_and_apply_grad(net_d.parameters(), grad_d_fake, 0., arg.clip_norm, device,
                      replace_grad=False)
  err_d_item = 0
  optimizer_d.step()

  ############################
  # (2) Update G network: maximize log(D(G(z)))
  ###########################
  if global_step % arg.train_gen_every_n_steps == 0:
    net_d.zero_grad()
    net_g.zero_grad()
    label.fill_(REAL_LABEL)  # fake labels are real for generator cost
    # fake = net_g(pt.randn(b_size, arg.nz, 1, 1, device=device))
    fake_pred = net_d(fake, rand_labels).view(-1)
    err_g = criterion(fake_pred, label)
    err_g.backward()
    optimizer_g.step()


def main():
  parser = argparse.ArgumentParser()

  # PARAMS YOU LIKELY WANT TO SET
  parser.add_argument('--seed', type=int, default=999)
  parser.add_argument('--data', type=str, default='cifar10',
                      choices=['cifar10', 'celeba', 'imagenet32', 'celeba64', 'imagenet64'])
  parser.add_argument('--workers', type=int, default=2)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--batch_size_grad_acc', type=int, default=128)
  parser.add_argument('--nc', type=int, default=3)
  parser.add_argument('--nz', type=int, default=100)
  parser.add_argument('--ngf', type=int, default=64)
  parser.add_argument('--ndf', type=int, default=64)

  parser.add_argument('--n_epochs', type=int, default=20)
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
  parser.add_argument('--pretrain_exp', type=str, default=None)
  parser.add_argument('--single_iter', action='store_true')
  parser.add_argument('--local_data', action='store_true')
  parser.add_argument('--labeled', action='store_true')

  arg = parser.parse_args()

  save_dir = os.path.join('../logs/dcgan/', arg.exp_name)

  os.makedirs(save_dir, exist_ok=True)
  print("Random Seed: ", arg.seed)
  random.seed(arg.seed)
  pt.manual_seed(arg.seed)
  if arg.labeled:
    assert arg.model == 'resnet'
    assert arg.data in {'cifar10', 'imagenet32'}

  if arg.target_eps is not None:
    n_samples_dict = {'cifar10': 50_000, 'imagenet32': 1_281_167, 'celeba': 202_599,
                      'imagenet64': 1_281_167, 'celeba64': 202_599}
    n_samples = n_samples_dict[arg.data]
    n_iter = np.ceil(n_samples / arg.batch_size) * arg.n_epochs
    actual_eps, noise_factor = find_dpsgd_sigma(arg.target_eps, arg.target_delta, arg.batch_size,
                                                n_samples, n_iter)
    print(f'using sigma={noise_factor} to achieve eps={actual_eps}')
  else:
    noise_factor = None

  data_root = '/tmp' if arg.local_data else '../data'
  img_hw = 32
  if arg.data == 'cifar10':
    # dataloader = load_cifar10('../data', arg.batch_size, arg.workers)
    dataloader, _ = load_cifar10(32, data_root, arg.batch_size, arg.workers, 'normed', True, False)
    n_classes = 10 if arg.labeled else None
  elif arg.data == 'celeba':
    dataloader = load_celeba(data_root, arg.batch_size, arg.workers)
    n_classes = None
  elif arg.data == 'celeba64':
    dataloader = load_celeba(data_root, arg.batch_size, arg.workers,
                             image_size=64, center_crop_size=64)
    n_classes = None
    img_hw = 64
    arg.data = 'celeba'
  elif arg.data == 'imagenet64':
    dataloader = load_imagenet_numpy(data_root, arg.batch_size, arg.workers, img_hw=64)
    n_classes = 1000 if arg.labeled else None
    img_hw = 64
  elif arg.data == 'imagenet32':
    dataloader = load_imagenet_numpy(data_root, arg.batch_size, arg.workers, img_hw=32)
    n_classes = 1000 if arg.labeled else None
  else:
    raise ValueError

  device = pt.device("cuda:0" if (pt.cuda.is_available() and arg.n_gpu > 0) else "cpu")

  if arg.model == 'resnet':

    net_d, net_g = get_resnets(arg.nz, arg.ngf, arg.ndf, n_classes, device, img_hw)
  else:
    assert img_hw == 32
    net_d, net_g = get_convnets(arg, device)

  if arg.pretrain_exp is not None:
    weights_file = os.path.join('../logs/dcgan/', arg.pretrain_exp, 'ck.pt')
    pretrained_weights = pt.load(weights_file)
    net_g.load_state_dict(pretrained_weights['g'])
    net_d.load_state_dict(pretrained_weights['d'])
    del pretrained_weights

  criterion = nn.BCEWithLogitsLoss()
  if arg.labeled:
    def disc_loss(x, t, y):
      return criterion(net_d(x, t).view(-1), y)
  else:
    def disc_loss(x, y):
      return criterion(net_d(x).view(-1), y)

  # Create batch of latent vectors that we will use to visualize
  #  the progression of the generator

  # Establish convention for real and fake labels during training

  # Setup Adam optimizers for both G and D
  lr_dis = arg.lr_default if arg.lr_dis is None else arg.lr_dis
  lr_gen = arg.lr_default if arg.lr_gen is None else arg.lr_gen

  if arg.clip_norm is not None:
    net_d = extend(net_d)
    criterion = extend(criterion)
    # net_g = extend(net_g, use_converter=True)

  optimizer_d = optim.Adam(net_d.parameters(), lr=lr_dis, betas=(arg.beta1, 0.999))

  optimizer_g = optim.Adam(net_g.parameters(), lr=lr_gen, betas=(arg.beta1, 0.999))

  # Training Loop

  # Lists to keep track of progress
  # img_list = []
  g_losses = []
  d_losses = []
  iters = 0

  print("Starting Training Loop...")
  global_step = 0
  err_g = None
  real_pred_acc, fake_pred_acc = None, None
  for epoch in range(arg.n_epochs):
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
      if not arg.labeled:
        if arg.clip_norm is None:
          res = nondp_unlabeled_update(net_d, net_g, optimizer_d, optimizer_g, real_batch, criterion,
                                       disc_label, arg, b_size, device,
                                       global_step)
          real_pred_acc, fake_pred_acc, err_d_item, err_g = res
        else:  # PRIVATE SETTING
          dp_unlabeled_update(net_d, net_g, optimizer_d, optimizer_g, real_batch, criterion,
                              disc_loss, disc_label, arg, b_size, device,
                              global_step, noise_factor)
          err_d_item = 0
      else:
        real_labels = data[1].to(device)
        if arg.clip_norm is None:
          res = nondp_labeled_update(net_d, net_g, optimizer_d, optimizer_g, real_batch, criterion,
                                     disc_label, arg, b_size, device,
                                     global_step, real_labels, n_classes)
          real_pred_acc, fake_pred_acc, err_d_item, err_g = res
        else:  # PRIVATE SETTING
          dp_labeled_update(net_d, net_g, optimizer_d, optimizer_g, real_batch, criterion,
                            disc_loss, disc_label, arg, b_size, device,
                            global_step, noise_factor, real_labels, n_classes)
          err_d_item = 0

      err_g_item = 0 if err_g is None else err_g.item()
      # Output training stats
      if i % 100 == 0:
        if real_pred_acc is not None:
          print(f'real_pred_acc {real_pred_acc}, fake pred acc {fake_pred_acc}')
        else:
          print(f'err_d {err_d_item}, err_g {err_g_item}')
      #   print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
      #         % (epoch, arg.n_epochs, i, len(dataloader),
      #            err_d_item, err_g_item, D_x, D_G_z1, D_G_z2))

      # if i % 500 == 0:
      #   syn_data_path = create_fid_dataset(net_g, arg.nz, device, save_dir,
      #                                      file_name=f'synth_data_{i}')
      #   fid = get_fid_scores(syn_data_path, dataset_name=arg.data, device=device, n_samples=50_000,
      #                        image_size=32, center_crop_size=32, data_scale='normed',
      #                        batch_size=100)
      #   print(f'fid at it {i}: {fid}')
      #   os.remove(syn_data_path)

      # Save Losses for plotting later
      g_losses.append(err_g_item)
      d_losses.append(err_d_item)

      iters += 1

      if arg.single_iter:
        break
    if arg.single_iter:
      break

  if arg.labeled:
    assert n_classes == 10
    plot_labels = pt.eye(10, device=device)[pt.repeat_interleave(pt.arange(0, 10), 10)]
    labeled_noise, rand_labels = get_gen_noise(100, arg.nz, device, 10, plot_labels)
    fake = net_g(labeled_noise).detach().cpu()
  else:
    fake = net_g(get_gen_noise(100, arg.nz, device)).detach().cpu()
  if arg.data in {'cifar10', 'celeba'}:
    mean_tsr = pt.tensor([0.485, 0.456, 0.406], device=fake.device)
    sdev_tsr = pt.tensor([0.229, 0.224, 0.225], device=fake.device)
  else:
    mean_tsr = pt.tensor([0.5, 0.5, 0.5], device=fake.device)
    sdev_tsr = pt.tensor([0.5, 0.5, 0.5], device=fake.device)
  data_to_print = fake * sdev_tsr[None, :, None, None] + mean_tsr[None, :, None, None]
  data_to_print = pt.clamp(data_to_print, min=0., max=1.)
  img_path_clamp = os.path.join(save_dir, 'clamped_plot.png')
  vutils.save_image(data_to_print, img_path_clamp, normalize=True, nrow=10)
  final_images = vutils.make_grid(fake, padding=2, normalize=True, scale_each=True)

  pt.save({'fake': fake}, os.path.join(save_dir, 'fake_final.pt'))
  pt.save({'d': net_d.state_dict(), 'g': net_g.state_dict()}, os.path.join(save_dir, 'ck.pt'))
  syn_data_path = create_fid_dataset(net_g, arg.nz, device, save_dir, file_name='synth_data')
  fid = get_fid_scores(syn_data_path, dataset_name=arg.data, device=device, n_samples=50_000,
                       image_size=img_hw, center_crop_size=img_hw, data_scale='normed', batch_size=100)
  print(f'fid={fid}')
  np.save(os.path.join(save_dir, 'fid.npy'), fid)
  os.remove(syn_data_path)
  make_final_plots(dataloader, g_losses, d_losses, device, final_images, save_dir)


if __name__ == '__main__':
  main()
