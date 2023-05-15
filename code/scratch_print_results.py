import os

import numpy as np


def dpmerf_cifar_results():
  print("fid")
  avg_fids = []
  std_fids = []
  avg_accuracies = []
  std_accuracies = []
  temp_sum = []
  sum_count = 0
  for idx in range(18):
    res_file = f'../logs/dpmerf/may17_dpmerf_res/run_{idx}/fid.npy'
    fid = np.load(res_file)
    print(f'|run {idx} | {fid:6.2f} |')
    temp_sum.append(fid)
    sum_count += 1
    if sum_count == 3:
      avg_fids.append(np.mean(temp_sum))
      std_fids.append(np.std(temp_sum))
      sum_count = 0
      temp_sum = []

  print('mean fids')
  print('mean fids')
  for mean, std in zip(avg_fids, std_fids):
    print(f'mean {mean}, std {std}')

  print('acc')
  for idx in range(18):
    res_file = f'../logs/dpmerf/may17_dpmerf_res/run_{idx}/accuracies.npz'
    test_acc = np.load(res_file)['test_acc']
    print(f'|run {idx} | {test_acc:6.2f} |')
    temp_sum.append(test_acc)
    sum_count += 1
    if sum_count == 3:
      avg_accuracies.append(np.mean(temp_sum))
      std_accuracies.append(np.std(temp_sum))
      sum_count = 0
      temp_sum = []

  print('mean test accs')
  for mean, std in zip(avg_accuracies, std_accuracies):
    print(mean, std)


def dcgan_results():
  # n_cifar_exp = 432
  # print('mar15_dcgan_celeba64_wide_search fids:')
  # for idx in range(n_cifar_exp):
  #   fid_file = f'../logs/dcgan/mar15_dcgan_celeba64_wide_search/run_{idx}/fid.npy'
  #   if os.path.exists(fid_file):
  #     score = np.load(fid_file)
  #     if score < 100.:
  #       print(f'run {idx}: {score}')
  # n_cifar_exp = 72
  # print('mar15_dcgan_celeba64_param_search_smaller_net fids:')
  # for idx in range(n_cifar_exp):
  #   fid_file = f'../logs/dcgan/mar15_dcgan_celeba64_param_search_smaller_net/run_{idx}/fid.npy'
  #   if os.path.exists(fid_file):
  #     print(f'run {idx}: {np.load(fid_file)}')
  # n_cifar_exp = 150
  # print('mar8_dcgan_celeba64 fids:')
  # for idx in range(n_cifar_exp):
  #   fid_file = f'../logs/dcgan/mar8_dcgan_celeba64_nondp/run_{idx}/fid.npy'
  #   if os.path.exists(fid_file):
  #     print(f'run {idx}: {np.load(fid_file)}')
  # n_cifar_exp = 72
  # print('dcgan debug exp fids:')
  # for idx in range(n_cifar_exp):
  #   fid_file = f'../logs/dcgan/oct13_dpgan_celeba_nondp_local/run_{idx}/fid.npy'
  #   if os.path.exists(fid_file):
  #     print(f'run {idx}: {np.load(fid_file)}')
  #
  # n_cifar_exp = 64
  # print('dcgan debug exp fids:')
  # for idx in range(n_cifar_exp):
  #   fid_file = f'../logs/dcgan/oct17_dcgan_celeba/run_{idx}/fid.npy'
  #   if os.path.exists(fid_file):
  #     print(f'run {idx}: {np.load(fid_file)}')

  n_cifar_exp = 30
  acc = []
  print('mar16_dcgan_celeba64_prelim_eps05_2_10 fids:')
  for idx in range(n_cifar_exp):
    fid_file = f'../logs/dcgan/mar16_dcgan_celeba64_prelim_eps05_2_10/run_{idx}/fid.npy'
    if os.path.exists(fid_file):
      fid = np.load(fid_file)
      print(f'run {idx}: {fid}')
      acc.append(fid)
    else:
      acc.append(None)
    if len(acc) == 5:
      if None in acc:
        print('result incomplete')
      else:
        print(np.mean(acc), np.std(acc))
      acc = []

  n_cifar_exp = 50
  acc = []
  print('mar16_dcgan_celeba64_pre_prelim_eps02_1_5 fids:')
  for idx in range(n_cifar_exp):
    fid_file = f'../logs/dcgan/mar16_dcgan_celeba64_pre_prelim_eps02_1_5/run_{idx}/fid.npy'
    if os.path.exists(fid_file):
      fid = np.load(fid_file)
      print(f'run {idx}: {fid}')
      acc.append(fid)
    else:
      acc.append(None)
    if len(acc) == 5:
      if None in acc:
        print('result incomplete')
      else:
        print(np.mean(acc), np.std(acc))
      acc = []


if __name__ == '__main__':
  # dpmerf_cifar_results()
  dcgan_results()
