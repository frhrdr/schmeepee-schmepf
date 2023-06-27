import os
import argparse
import numpy as np
from collections import OrderedDict


def load_best_fid(run_dir):
  saved_checkpoints = [a for a in os.listdir(run_dir) if a.startswith('checkpoint_ep')]
  if len(saved_checkpoints) < 1:
    print(f'best checkpoint unclear for {run_dir}')
    return -1, -1
  best_fid_ep = int(saved_checkpoints[0][len('checkpoint_ep'):-len('.pt')])
  best_fid_file = f'fid_ep{best_fid_ep}.npy'
  try:
    fid = np.load(os.path.join(run_dir, best_fid_file))
  except FileNotFoundError as fne:
    print(f'File not found: {fne}')
    fid = -1
  return fid, best_fid_ep


def load_sorted_fids(run_dir):
  saved_fids = sorted([a for a in os.listdir(run_dir) if a.startswith('fid_ep')])
  ordered_fids = OrderedDict()
  for fid_file in saved_fids:
    fid_ep = int(fid_file[len('fid_ep'):-len('.npy')])
    try:
      fid = np.load(os.path.join(run_dir, fid_file))
    except FileNotFoundError as fne:
      print(f'File not found: {fne}')
      fid = -1
    ordered_fids[fid_ep] = fid
  return ordered_fids


def dcgan_results():
  parser = argparse.ArgumentParser()

  # PARAMS YOU LIKELY WANT TO SET
  parser.add_argument('--base_logdir', default='../logs/dcgan')
  parser.add_argument('--logdir', '-l')
  parser.add_argument('--n_runs', type=int, default=1)
  parser.add_argument('--avgn', type=int, default=1)
  arg = parser.parse_args()

  best_result = None
  best_run = None
  avg_acc = []
  for idx in range(arg.n_runs):
    run_dir = f'../logs/dcgan/{arg.logdir}/run_{idx}/'
    # find best fid epoch
    run_best_fid, run_best_fid_ep = load_best_fid(run_dir)
    sorted_fids = load_sorted_fids(run_dir)
    sorted_fid_str = ', '.join([f'{ep}:{fid:.2f}' for (ep, fid) in sorted_fids.items()])
    print(f'run {idx} best fid at ep {run_best_fid_ep}={run_best_fid:.4f}  all fids={sorted_fid_str}')
    avg_acc.append(run_best_fid)
    if 1 < arg.avgn == len(avg_acc):
      if None in avg_acc:
        print('result incomplete')
      else:
        print(f'average over {arg.avgn} runs: mean={np.mean(avg_acc):.4f}, sdev={np.std(avg_acc):.4f}')
      avg_acc = []
    if best_result is None or 0 < run_best_fid < best_result:
      best_result = run_best_fid
      best_run = idx

  print(f'best overall result is FID={best_result:.4f} at run {best_run}')


if __name__ == '__main__':
  # dpmerf_cifar_results()
  dcgan_results()
