import os
import argparse
import numpy as np
from util_logging import mnist_synth_to_real_test
from torchvision import utils as vutils
import torch as pt
import shutil
from eval_accuracy import synth_to_real_test
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.models as tvmodels

def debug_mnist_eval():
  dataset = 'dmnist'
  data_path = '../logs/may6_dmnist_nondp_lr_gridsearch/run_1/synth_data_it350000.npz'
  writer = None
  log_dir = '.'
  acc_file_name = 'dmnist-debug-acc'
  step = 1
  mnist_synth_to_real_test(dataset, data_path, writer, log_dir, acc_file_name, step)


def plot_data():
  data_file = '../logs/may12_cifar_nondp_labeled_gridsearch/run_1/synth_data.npz'
  data_dict = np.load(data_file)
  data = data_dict['x']
  labels = data_dict['y']
  perm = np.random.permutation(data.shape[0])[:100]
  imgs = pt.tensor(data[perm])
  print(labels[perm])
  vutils.save_image(imgs, 'dataset-debug-cifar.png', normalize=True, nrow=10)
  # acc_file = '../logs/may10_dmnist_debug/run_4/accuracies_it4000.npz'
  # accs = np.load(acc_file)
  # for k, v in accs.items():
  #   print(f'{k}: {v}')


def make_imagenet_subset():
  n_subset = 300
  n_samples = 100_000
  testset_path = '/home/fharder/imagenet/imgnet/test'
  picks = np.random.permutation(n_samples)[:n_subset]
  subset_path = f'/home/fharder/imagenet/test_subset_{n_subset}/'
  os.makedirs(subset_path, exist_ok=False)
  for pick in picks:
    img_name = f'ILSVRC2012_test_{pick:08d}.JPEG'
    shutil.copyfile(os.path.join(testset_path, img_name),
                    os.path.join(subset_path, img_name))


def redo_all_classifications():
  redo_classification('may13_cifar_nondp_labeled_res', 'synth_data')
  redo_classification('may13_cifar_labeled_dp_res', 'synth_data')
  redo_classification('dpmerf/may17_dpmerf_res', 'gen_data')


def redo_classification(exp_name, data_file_str):
  device = 0
  base_dir = '../logs/'
  # exp_name = 'may13_cifar_nondp_labeled_res'  # 'may13_cifar_labeled_dp_res'
  # exp_name = 'may13_cifar_labeled_dp_res'  # 'may13_cifar_labeled_dp_res'
  # exp_name = 'dpmerf/may17_dpmerf_res'
  # data_file_str = 'synth_data'
  # data_file_str = 'gen_data'
  log_dir = os.path.join(base_dir, exp_name)
  run_dirs = [d for d in os.scandir(log_dir) if (d.is_dir() and d.name.startswith('run'))]
  for run in run_dirs:
    syn_data_files = [f for f in os.scandir(run.path) if f.name.startswith(data_file_str)]
    saved_accs_dir = os.path.join(run.path, 'accs_with_errors')
    os.makedirs(saved_accs_dir, exist_ok=True)

    for data_file in syn_data_files:
      if data_file.name == f'{data_file_str}.npz':
        acc_file_name = 'accuracies.npz'
      else:
        assert data_file.name.startswith(f'{data_file_str}_it') and data_file.name.endswith('.npz')
        acc_file_name = f"accuracies_it{data_file.name[len('synth_data_it'):]}"
      acc_file_path = os.path.join(run.path, acc_file_name)
      if os.path.exists(acc_file_path):
        shutil.copyfile(acc_file_path, os.path.join(saved_accs_dir, acc_file_name))
      print(f'redoing eval for: {data_file.path}')
      test_acc, train_acc = synth_to_real_test(device, data_file.path, scale_to_range=True)
      print(f'new test acc: {test_acc}')
      np.savez(acc_file_path, test_acc=test_acc, train_acc=train_acc)


def plot_pruning_loss_exp():
  # log_path = '../logs/may17_cifar_pruned_loss_nondp_eval/pruned_loss_comp.npz'
  log_path = 'pruned_loss_comp.npz'
  log = np.load(log_path)
  xrange = np.arange(len(log['pruned_losses'])) * 100
  plt.plot(xrange, log['pruned_losses'], label='optimized feature loss')
  plt.plot(xrange, log['inv_pruned_losses'], label='pruned feature loss')
  plt.legend()
  plt.savefig('pruned_loss_comp.png')


def dpmerf_cifar_results():
  print("fid")
  avg_fids = []
  avg_accuracies = []
  temp_sum = 0
  sum_count = 0
  for idx in range(18):
    res_file = f'../logs/dpmerf/may17_dpmerf_res/run_{idx}/fid.npy'
    fid = np.load(res_file)
    print(f'|run {idx} | {fid:6.2f} |')
    temp_sum += fid
    sum_count += 1
    if sum_count == 3:
      avg_fids.append(temp_sum / 3)
      sum_count = 0
      temp_sum = 0

  print("acc")
  for idx in range(18):
    res_file = f'../logs/dpmerf/may17_dpmerf_res/run_{idx}/accuracies.npy'
    test_acc = np.load(res_file)['test_acc']
    print(f'|run {idx} | {fid:6.2f} |')
    temp_sum += test_acc
    sum_count += 1
    if sum_count == 3:
      avg_accuracies.append(temp_sum / 3)
      sum_count = 0
      temp_sum = 0

    print('mean fids')
    for fid in avg_fids:
      print(fid)

    print('mean test accs')
    for acc in avg_accuracies:
      print(acc)


def get_celeba_imgs():
  transformations = [transforms.Resize(32), transforms.CenterCrop(32),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                     ]


  # folder dataset
  dataset = dset.ImageFolder(root='../data/img_align_celeba',
                             transform=transforms.Compose(transformations))
  dataloader = pt.utils.data.DataLoader(dataset, batch_size=100,
                                        shuffle=True, num_workers=1)
  for img, _ in dataloader:
    vutils.save_image(img, 'cebela_samples.png', normalize=True, nrow=10)
    break


def get_cifar_imgs():
  transformations = [transforms.Resize(32), transforms.ToTensor()]

  transformations.append(transforms.Lambda(lambda x: x*2 - 1))

  dataset = dset.CIFAR10(root='../data', train=True, download=False,
                         transform=transforms.Compose(transformations))

  dataloader = pt.utils.data.DataLoader(dataset, batch_size=1,
                                        shuffle=True, num_workers=1)
  n_loaded_per_class = [0] * 10
  imgs = [[] for _ in range(10)]
  for img, label in dataloader:
    if n_loaded_per_class[label] < 10:
      imgs[label].append(img)
      n_loaded_per_class[label] += 1
    else:
      if min(n_loaded_per_class) == 10:
        break
  imgs = pt.cat([pt.cat(k) for k in imgs])
  print(imgs.shape)
  vutils.save_image(imgs, 'cifar_samples.png', normalize=True, nrow=10)


def get_mnist_imgs():
  transformations = transforms.ToTensor()

  fmnist = dset.FashionMNIST('../data', train=True, transform=transformations, download=True)
  dmnist = dset.MNIST('../data', train=True, transform=transformations, download=True)

  def foo(dataset, name):
    dataloader = pt.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    n_loaded_per_class = [0] * 10
    imgs = [[] for _ in range(10)]
    for img, label in dataloader:
      if n_loaded_per_class[label] < 10:
        imgs[label].append(img)
        n_loaded_per_class[label] += 1
      else:
        if min(n_loaded_per_class) == 10:
          break
    imgs = pt.cat([pt.cat(k) for k in imgs])
    print(imgs.shape)
    vutils.save_image(imgs, f'{name}_samples.png', normalize=True, nrow=10)

  foo(fmnist, 'fmnist')
  foo(dmnist, 'dmnist')


def get_img_ranges():

  def find_min_max(dloader):
    glob_max = -100.
    glob_min = 100.
    for img, _ in dloader:
      imax = pt.max(img)
      imin = pt.min(img)
      if imax > glob_max:
        glob_max = imax
      if imin < glob_min:
        glob_min = imin
    return glob_min, glob_max

  transformations = [transforms.Resize(32), transforms.CenterCrop(32),
                     transforms.ToTensor()]

  # folder dataset
  dataset = dset.ImageFolder(root='../data/img_align_celeba',
                             transform=transforms.Compose(transformations))
  dataloader = pt.utils.data.DataLoader(dataset, batch_size=500,
                                        shuffle=True, num_workers=1)
  celeba_min, celeba_max = find_min_max(dataloader)
  print(f'celeba min={celeba_min}, max={celeba_max}')

  transformations = [transforms.Resize(32), transforms.ToTensor()]

  dataset = dset.CIFAR10(root='../data', train=True, download=False,
                         transform=transforms.Compose(transformations))

  dataloader = pt.utils.data.DataLoader(dataset, batch_size=500,
                                        shuffle=True, num_workers=1)
  cifar_min, cifar_max = find_min_max(dataloader)
  print(f'cifar min={cifar_min}, max={cifar_max}')

  transformations = transforms.ToTensor()
  dmnist = dset.MNIST('../data', train=True, transform=transformations, download=True)
  dataloader = pt.utils.data.DataLoader(dmnist, batch_size=1000, shuffle=True, num_workers=1)
  dmnist_min, dmnist_max = find_min_max(dataloader)
  print(f'dmnist min={dmnist_min}, max={dmnist_max}')

  fmnist = dset.FashionMNIST('../data', train=True, transform=transformations, download=True)
  dataloader = pt.utils.data.DataLoader(fmnist, batch_size=1000, shuffle=True, num_workers=1)
  fmnist_min, fmnist_max = find_min_max(dataloader)
  print(f'fmnist min={fmnist_min}, max={fmnist_max}')


def print_torchvision_models():
  # print('vgg16')
  # print(tvmodels.vgg16())
  # print('vgg13')
  # print(tvmodels.vgg13())
  # enc = tvmodels.convnext_tiny()
  # cnb_block_ids = [(1,0), (1,1), (1,2),
  #                  (3,0), (3,1), (3,2),
  #                  (5,0), (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), (5,7), (5,8)]
  # cnb_block_out_ids = [0, 3, 5]
  # sequential_conv_ids = [(0,0), (2, 1), (4, 1), (6, 1)]
  # conv_layers = [enc.features[0][0],
  #                enc.features[1][0].block[0], enc.features[1][0].block[5]]
  # print(conv_layers)
  print('inception')
  a = tvmodels.inception_v3()
  # print(a)
  # print(a.avgpool.)
  # print('efficientnetv2 m')
  # print(tvmodels.efficientnet_v2_m())
  # print('efficientnetv2 l')
  # print(tvmodels.efficientnet_v2_l())


def open_imagenet32():
  # for data_id in range(1, 11):
  #   data = np.load(f'../data/Imagenet32_train_npz/train_data_batch_{data_id}.npz')
  #   np.save(f'../data/Imagenet32_train_npz/train_data_batch_{data_id-1}_x.npy', data['data'])
  #   np.save(f'../data/Imagenet32_train_npz/train_data_batch_{data_id-1}_y.npy', data['labels']-1)
  #   if data_id == 1:
  #     np.save(f'../data/Imagenet32_train_npz/train_data_means.npy',
  #             data['mean'])
  #     break
    # print(f'batch {data_id}: shape {data["data"].shape}, mean start {data["mean"][:5]}')
    # n_classes = 1000
    # c_count = []
    # for c in range(1, n_classes + 1):
    #   c_count.append(np.sum(data['labels'] == c))
    # c_sorted = sorted(c_count)
    # print(f'batch {data_id}: classes min {c_sorted[:10]}, max {c_sorted[-10:]}')

  data = np.load(f'../data/Imagenet32_train_npz/train_data_batch_1_x.npy')
  # labels = np.load(f'../data/Imagenet32_train_npz/train_data_batch_1_y.npy')
  # print(labels[0], labels[-1], labels.dtype)
  # labels = np.memmap(f'../data/Imagenet32_train_npz/train_data_batch_1_y.npy', mode='r',
  #                    shape=(128116,), dtype=np.int64, order='C', offset=128)
  # print(labels[0], labels[-1])

  # data = np.memmap(f'../data/Imagenet32_train_npz/train_data_batch_1_x.npy', mode='r', shape=(128116, 3, 32, 32), order='C', offset=128)
  images = pt.tensor(data[:64, :].copy().reshape(64, 3, 32, 32))
  print(pt.max(images), pt.min(images), print(data.dtype))
  plt.figure(figsize=(8, 8))
  plt.axis("off")
  plt.title("Training Images")
  plt.imshow(
    np.transpose(vutils.make_grid(images, padding=2, normalize=False).cpu(),
                 (1, 2, 0)))
  plt.savefig('mar3_imagenet32samples.png')
  # from dcgan_baseline import Imagenet32Dataset
  # Imagenet32Dataset.npz_to_npy_batches()
  # pass

def dp_analysis_debug():
  niter = 391
  sigma = .53
  gamma = 128/50_000
  delta = 1e-6
  from autodp.autodp_core import Mechanism
  from autodp import mechanism_zoo, transformer_zoo

  class NoisySGD_mech(Mechanism):
    def __init__(self, prob, sigma, niter, name='NoisySGD'):
      Mechanism.__init__(self)
      self.name = name
      self.params = {'prob': prob, 'sigma': sigma, 'niter': niter}

      # create such a mechanism as in previously
      subsample = transformer_zoo.AmplificationBySampling()  # by default this is using poisson sampling
      mech = mechanism_zoo.GaussianMechanism(sigma=sigma)
      prob = prob
      # Create subsampled Gaussian mechanism
      SubsampledGaussian_mech = subsample(mech, prob, improved_bound_flag=True)

      # Now run this for niter iterations
      compose = transformer_zoo.Composition()
      mech = compose([SubsampledGaussian_mech], [niter])

      # Now we get it and let's extract the RDP function and assign it to the current mech being constructed
      rdp_total = mech.RenyiDP
      self.propagate_updates(rdp_total, type_of_update='RDP')

  noisysgd = NoisySGD_mech(prob=gamma, sigma=sigma, niter=niter)

  # compute epsilon, as a function of delta
  res = noisysgd.get_approxDP(delta=delta)
  print(res)


def cov_test():
  n = 10
  a = np.random.normal(loc=0, scale=1.0, size=(n, 4))

  mu = np.mean(a, axis=0)
  sigma = np.cov(a, rowvar=False)
  a_c = a - mu
  sigma_2 = (a_c.T) @ a_c / (n - 1)
  sigma_3 = np.zeros((4, 4))
  for i in range(n):
    sigma_3 += a_c[i:i+1, :].T @ a_c[i:i+1, :]
  sigma_3 /= n - 1
  print(sigma - sigma_3)


if __name__ == '__main__':
  pass
  # print_torchvision_models()
  # open_imagenet32()
  # dp_analysis_debug()
  # cov_test()
