import os.path

import torch as pt

import torchvision.transforms as transforms
import torchvision.datasets as dset
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_SDEV = [0.229, 0.224, 0.225]
CIFAR10_MEAN = [0.4914, 0.482, 0.447]
CIFAR10_SDEV = [0.247, 0.243, 0.262]
CELEBA32_MEAN = [0.508, 0.421, 0.375]
CELEBA32_SDEV = [0.294, 0.271, 0.269]
CELEBA64_MEAN = [0.508, 0.422, 0.377]
CELEBA64_SDEV = [0.301, 0.279, 0.277]


def shift_data_transform(x):
  return 2 * x - 1


def scale_transform(data_scale):
  if data_scale == 'normed':
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_SDEV)
  elif data_scale == 'bounded':
    return transforms.Lambda(shift_data_transform)
  elif data_scale == 'normed05':
    return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  elif data_scale == '0_1':
    return transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
  else:
    raise ValueError


def load_dataset(dataset_name, image_size, center_crop_size, dataroot, batch_size,
                 n_workers, data_scale, labeled=False, test_set=False):

  if dataset_name in ['celeba']:
    n_classes = None
    transformations = []
    if center_crop_size > image_size:
      transformations.extend([transforms.CenterCrop(center_crop_size),
                              transforms.Resize(image_size)])
    else:
      transformations.extend([transforms.Resize(image_size),
                              transforms.CenterCrop(center_crop_size)])

    transformations.extend([transforms.ToTensor(), scale_transform(data_scale)])

    # folder dataset
    dataset = dset.ImageFolder(root=os.path.join(dataroot, 'img_align_celeba'),
                               transform=transforms.Compose(transformations))
  elif dataset_name == 'lsun':
    n_classes = None
    transformations = []
    if center_crop_size > image_size:
      transformations.extend([transforms.CenterCrop(center_crop_size),
                              transforms.Resize(image_size)])
    else:
      transformations.extend([transforms.Resize(image_size),
                              transforms.CenterCrop(center_crop_size)])

    transformations.extend([transforms.ToTensor(), scale_transform(data_scale)])

    dataset = dset.LSUN(os.path.join(dataroot, 'lsun'), classes=['bedroom_train'],
                        transform=transforms.Compose(transformations))

  elif dataset_name == 'cifar10':
    return load_cifar10(image_size, dataroot, batch_size, n_workers, data_scale, labeled, test_set)

  elif dataset_name == 'stl10':
    n_classes = None
    transformations = [transforms.Resize(image_size), transforms.ToTensor(),
                       scale_transform(data_scale)]

    dataset = dset.STL10(root=dataroot, split='unlabeled', download=True,
                         transform=transforms.Compose(transformations))
  elif dataset_name in {'fmnist', 'dmnist', 'svhn', 'cifar10_pretrain'}:
    dataset = small_data_loader(dataset_name, not test_set)
    n_classes = 10
  elif dataset_name == 'imagenet':
    # dataset = load_imagenet_subset(center_crop_size, image_size, dataroot, data_scale)
    dataset = load_imagenet_val_set(center_crop_size, image_size, dataroot, data_scale)
    n_classes = 1000
  else:
    raise ValueError(f'{dataset_name} not recognized')

  assert dataset
  assert not test_set or dataset == 'cifar10'
  if labeled:
    assert n_classes is not None, 'selected dataset has no labels'
  else:
    n_classes = None

  dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=int(n_workers))

  return dataloader, n_classes


def load_imagenet_subset(center_crop_size, image_size, dataroot, data_scale):
  transformations = []

  if center_crop_size > image_size:
    transformations.extend([transforms.CenterCrop(center_crop_size),
                            transforms.Resize(image_size)])
  else:
    transformations.extend([transforms.Resize(image_size),
                            transforms.CenterCrop(center_crop_size)])

  transformations.extend([transforms.ToTensor(), scale_transform(data_scale)])

  # folder dataset
  dataset = dset.ImageFolder(root=os.path.join(dataroot, 'imagenet'),
                             transform=transforms.Compose(transformations))
  return dataset


def load_imagenet_val_set(center_crop_size, image_size, dataroot, data_scale):
  transformations = []

  if center_crop_size > image_size:
    transformations.extend([transforms.CenterCrop(center_crop_size),
                            transforms.Resize(image_size)])
  else:
    transformations.extend([transforms.Resize(image_size),
                            transforms.CenterCrop(center_crop_size)])

  transformations.extend([transforms.ToTensor(), scale_transform(data_scale)])

  # folder dataset
  dataset = dset.ImageFolder(root=os.path.join(dataroot, 'imagenet_val'),
                             transform=transforms.Compose(transformations))
  return dataset


def load_cifar10(image_size, dataroot, batch_size,
                 n_workers, data_scale, labeled=False, test_set=False):
  transformations = [transforms.Resize(image_size), transforms.ToTensor(),
                     scale_transform(data_scale)]
  dataset = dset.CIFAR10(root=dataroot, train=not test_set, download=True,
                         transform=transforms.Compose(transformations))

  n_classes = 10 if labeled else None
  dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=int(n_workers))

  return dataloader, n_classes


def load_synth_dataset(data_file, batch_size, subset_size=None, to_tensor=False, shuffle=True,
                       source_data_scale=None, target_data_scale=None):
  if data_file.endswith('.npz'):  # allow for labels
    data_dict = np.load(data_file)
    data = data_dict['x']
    if 'y' in data_dict.keys():
      targets = data_dict['y']
      if len(targets.shape) > 1:
        targets = np.squeeze(targets)
        assert len(targets.shape) == 1, f'need target vector. shape is {targets.shape}'
    else:
      targets = None

    if subset_size is not None:
      random_subset = np.random.permutation(data_dict['x'].shape[0])[:subset_size]
      data = data[random_subset]
      targets = targets[random_subset] if targets is not None else None

    # revert scaling if necessary
    if target_data_scale is not None:
      print(f'rescaling data of shape {data.shape} from {source_data_scale} to {target_data_scale}')
      print(f'vals as loaded: {np.min(data)}, {np.max(data)}, {data.shape}')
      assert source_data_scale is not None
      assert target_data_scale == '0_1'
      if source_data_scale == 'normed':
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        sdev = np.asarray(IMAGENET_SDEV, dtype=np.float32)
      elif source_data_scale == 'bounded':
        mean = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
        sdev = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
      elif source_data_scale == 'normed05':
        mean = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
        sdev = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
      elif source_data_scale == '0_1':
        mean = np.asarray([0., 0., 0.], dtype=np.float32)
        sdev = np.asarray([1., 1., 1.], dtype=np.float32)
      else:
        raise ValueError
      data = data * sdev[None, :, None, None] + mean[None, :, None, None]
      print(f'vals as rescaled: {np.min(data)}, {np.max(data)}, {data.shape}')

    synth_data = SynthDataset(data=data, targets=targets, to_tensor=to_tensor)
  else:  # old version
    assert source_data_scale is None and target_data_scale is None
    data = np.load(data_file)
    if subset_size is not None:
      data = data[np.random.permutation(data.shape[0])[:subset_size]]
    synth_data = SynthDataset(data, targets=None, to_tensor=False)

  synth_dataloader = pt.utils.data.DataLoader(synth_data, batch_size=batch_size, shuffle=shuffle,
                                              drop_last=False, num_workers=1)
  return synth_dataloader


class SynthDataset(pt.utils.data.Dataset):
  def __init__(self, data, targets, to_tensor):
    self.labeled = targets is not None
    self.data = data
    self.targets = targets
    self.to_tensor = to_tensor

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    d = pt.tensor(self.data[idx], dtype=pt.float32) if self.to_tensor else self.data[idx]
    if self.labeled:
      t = pt.tensor(self.targets[idx], dtype=pt.long) if self.to_tensor else self.targets[idx]
      return d, t
    else:
      return d


def mnist_transforms(is_train):
  # if model == 'vgg15':
  #   transform_train = transforms.Compose([
  #     transforms.Resize((32, 32)),
  #     transforms.RandomCrop(32, padding=4),
  #     transforms.RandomHorizontalFlip(),
  #     transforms.ToTensor(),
  #     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
  #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  #   ])
  #   transform_test = transforms.Compose([
  #     transforms.Resize((32, 32)),
  #     transforms.ToTensor(),
  #     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
  #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  #   ])

  # elif model == 'resnet18':
  transform_train = transforms.Compose([
    transforms.CenterCrop((28, 28)),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.5], [0.5])
  ])
  transform_test = transforms.Compose([
    transforms.CenterCrop((28, 28)),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.5], [0.5])
  ])
  # else:
  #   raise ValueError
  return transform_train if is_train else transform_test


def small_data_loader(dataset_name, is_train):
  assert dataset_name in {'fmnist', 'dmnist', 'svhn', 'cifar10_pretrain'}

  if dataset_name in {'fmnist', 'dmnist'}:
    transform = mnist_transforms(is_train)
  else:
    transform = transforms.Compose([
      transforms.CenterCrop((28, 28)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

  root = '../data/'
  if dataset_name == 'dmnist':
    return dset.MNIST(root, train=is_train, transform=transform, download=True)
  elif dataset_name == 'fmnist':
    return dset.FashionMNIST(root, train=is_train, transform=transform, download=True)

  elif dataset_name == 'svhn':
    svhn_dir = os.path.join(root, 'svhn')
    os.makedirs(svhn_dir, exist_ok=True)
    train_str = 'train' if is_train else 'test'
    return dset.SVHN(svhn_dir, split=train_str, transform=transform, download=True)
  elif dataset_name == 'cifar10_pretrain':
    return dset.CIFAR10(root=root, train=is_train, transform=transform, download=True)
  else:
    raise ValueError


class Imagenet32Dataset(pt.utils.data.Dataset):
  def __init__(self, data_root, transform=None, test_set=False):
    assert not test_set
    self.data_subdir = os.path.join(data_root, f'Imagenet32_{"val" if test_set else "train"}_npz')
    self.default_batch_len = 128116
    self.last_batch_len = 128123
    self.n_feats = 3072
    self.mm_offset = 128
    self.memmaps = []
    self.mean = np.load(os.path.join(self.data_subdir, 'train_data_means.npy')).reshape((3, 32, 32))
    self.transform = transform
    for data_batch_id in range(10):
      len_batch = self.default_batch_len if data_batch_id < 9 else self.last_batch_len

      x_map = np.memmap(os.path.join(self.data_subdir, f'train_data_batch_{data_batch_id}_x.npy'),
                        dtype=np.uint8, mode='r', shape=(len_batch, 3, 32, 32),
                        offset=self.mm_offset)
      y_map = np.memmap(os.path.join(self.data_subdir, f'train_data_batch_{data_batch_id}_y.npy'),
                        dtype=np.uint8, mode='r', shape=(len_batch,),
                        offset=self.mm_offset)
      self.memmaps.append((x_map, y_map))

  @staticmethod
  def npz_to_npy_batches():
    for data_id in range(1, 11):
      data = np.load(f'../data/Imagenet32_train_npz/train_data_batch_{data_id}.npz')
      np.save(f'../data/Imagenet32_train_npz/train_data_batch_{data_id-1}_x.npy', data['data'])
      np.save(f'../data/Imagenet32_train_npz/train_data_batch_{data_id-1}_y.npy', data['labels']-1)
      if data_id == 1:
        np.save(f'../data/Imagenet32_train_npz/train_data_means.npy', data['mean'])

    data = np.load(f'../data/Imagenet32_val_npz/val_data.npz')
    np.save(f'../data/Imagenet32_val_npz/val_data_x.npy', data['data'])
    np.save(f'../data/Imagenet32_val_npz/val_data_y.npy', data['labels'] - 1)

  def __len__(self):
    return self.default_batch_len * 9 + self.last_batch_len

  def __getitem__(self, idx):
    batch_id = idx // self.default_batch_len
    sample_id = idx % self.default_batch_len
    if batch_id == 10:
      batch_id = 9
      sample_id += self.default_batch_len
    x_map, y_map = self.memmaps[batch_id]
    x, y = x_map[sample_id].copy(), y_map[sample_id].copy()
    x = pt.tensor(x, dtype=pt.float) / 255
    if self.transform:
      x = self.transform(x)
    return x, y


def load_imagenet_32(data_root, batch_size, workers, test_set=False):
  transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  # transform = None
  dataset = Imagenet32Dataset(data_root, transform, test_set)
  # Create the dataloader
  dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers)
  return dataloader
