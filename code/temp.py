import os
import torch as pt
import numpy as np


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
