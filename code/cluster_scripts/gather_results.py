import os
import numpy as np
import argparse
import pandas as pd
import torch as pt

def has_any_results(log_dir):
  run_dirs = [d for d in os.scandir(log_dir) if (d.is_dir() and d.name.startswith('run'))]
  for run in run_dirs:
    args_file = [f for f in os.scandir(run.path) if f.name == 'args.csv']
    eval_file = [f for f in os.scandir(run.path) if f.name == 'eval_scores.csv']
    if args_file and eval_file:
      return True
  return False


def get_rundirs(log_dir):
  run_dirs = [d for d in os.scandir(log_dir) if (d.is_dir() and d.name.startswith('run'))]
  if all([d.name.startswith('run_') for d in run_dirs]):
    sorted_dirs = sorted(run_dirs, key=lambda d: int(d.name[len('run_'):]))
    sorted_dirs = {int(d.name[len('run_'):]): d for d in sorted_dirs}
  else:
    sorted_dirs = sorted(run_dirs, key=lambda d: d.name)
    sorted_dirs = {d.name: d for d in sorted_dirs}
  return sorted_dirs


def experiment_table(log_dir):
  run_dirs = get_rundirs(log_dir)
  args_series = {}
  eval_tables = {}
  args_index = pd.Index([])
  eval_metrics = pd.Index([])
  eval_iters = pd.Index([])
  found_any_results = False
  for run_id, run_dir in run_dirs.items():
    args_file = os.path.join(run_dir, 'args.csv')
    eval_file = os.path.join(run_dir, 'eval_scores.csv')
    if os.path.exists(args_file):
      args_series[run_id] = pd.read_csv(args_file, index_col=0).squeeze('columns')
      args_index = args_index.union(args_series[run_id].index)
    else:
      args_series[run_id] = pd.DataFrame([])

    if os.path.exists(eval_file):
      eval_tables[run_id] = pd.read_csv(eval_file, index_col=0)
      eval_metrics = eval_metrics.union(eval_tables[run_id].index)
      eval_iters = eval_iters.union(eval_tables[run_id].columns)
      found_any_results = True
    else:
      eval_tables[run_id] = pd.DataFrame([])
  assert found_any_results
  # scores_then_args = eval_index.union(args_index)
  # runs_by_metric_index = pd.MultiIndex.from_product((eval_cols, scores_then_args))
  runs_by_metric_index = pd.MultiIndex.from_product((eval_iters, eval_metrics))
  args_multi_index = pd.MultiIndex.from_product((['args'], args_index))
  runs_by_metric_plus_args_index = runs_by_metric_index.union(args_multi_index)
  exp_table = pd.DataFrame(index=run_dirs.keys(), columns=runs_by_metric_plus_args_index)

  for run_id in run_dirs:
    run_args = args_series[run_id]
    run_eval = eval_tables[run_id]

    for arg in run_args.index:
      exp_table.loc[run_id, ('args', arg)] = run_args[arg]

    for eval_iter in run_eval.columns:
      eval_iter = str(eval_iter)
      for eval_metric in run_eval.index:
        vals = run_eval.loc[eval_metric, eval_iter]
        exp_table.loc[run_id, (eval_iter, eval_metric)] = vals

  return exp_table


def remove_matching_args(exp_table):
  first_row = exp_table.loc[0, 'args']
  matches = exp_table.loc[:, 'args'].eq(first_row, axis='columns')
  nan_cols = exp_table.loc[:, 'args'].isnull().all(axis='index')
  matches = matches.all(axis='index')
  matching_ids = [k for k in matches.index if (matches[k] or nan_cols[k])]
  reduced_exp_table = exp_table.drop(labels=[('args', k) for k in matching_ids], axis=1)
  return reduced_exp_table


def remove_redundant_args(exp_table, redundant_cols):
  # try to drop a couple of columns that are usually unnecessary
  cols_to_drop = [('args', k) for k in redundant_cols if (('args', k) in exp_table.columns)]
  reduced_exp_table = exp_table.drop(labels=cols_to_drop, axis=1)
  return reduced_exp_table


def final_iter_table(exp_table):
  # duplicate column entries may take the form '1000.x' and will be ignored
  exp_iters = [int(k[0]) for k in exp_table.columns if k[0].isdigit()]
  max_iter = max(exp_iters)
  return exp_table[[str(max_iter), 'args']]


def best_iter_table(exp_table, iter_idx=0):
  # duplicate column entries may take the form '1000.x' and will be ignored
  exp_iters = [int(k[iter_idx]) for k in exp_table.columns if k[iter_idx].isdigit()]
  max_iter = str(max(exp_iters))
  candidate_cols = ['best', 'proxy_best', max_iter, 'args']
  select_cols = [c for c in candidate_cols if c in exp_table.columns]
  best_table = exp_table[select_cols]
  best_table = best_table.rename(columns={max_iter: 'final', 'best': 'best_fid'})
  return best_table


def average_n_indices(exp_table, average_n):
  exp_table_floats = exp_table.drop('args', axis=1)
  table_grouped = exp_table_floats.groupby(np.arange(len(exp_table_floats)) // average_n)

  def new_idx(idx):
    return f'{idx*average_n}-{idx*average_n+average_n-1}'

  table_mean = table_grouped.mean()
  table_mean = table_mean.rename(index={k: new_idx(k) for k in table_mean.index})
  table_sdev = table_grouped.std()
  table_sdev = table_sdev.rename(index={k: new_idx(k) for k in table_sdev.index})

  def arg_match(x):
    first_row = x.loc[x.index[0]].copy()
    matches = x.eq(first_row, axis='columns').all(axis='index')
    for idx in matches.index:
      if not matches.loc[idx]:
        first_row.loc[idx] = np.nan

    args_table = pd.DataFrame(first_row).T
    args_table = args_table.rename(index={args_table.index[0]: 'dummy'})
    return args_table

  table_args = exp_table[['args']]
  table_args = table_args.groupby(np.arange(len(exp_table_floats)) // average_n).apply(arg_match)
  table_args = table_args.reorder_levels([1, 0], axis=0).loc['dummy'].copy()
  table_args = table_args.rename(index={k: new_idx(k) for k in table_args.index})

  cat_table = pd.concat({'mean': table_mean, 'sdev': table_sdev, 'args': table_args}, axis=1)
  cat_table = cat_table.reorder_levels([1, 0, 2], axis=1)
  return cat_table


def view_table(log_file, view_metrics=None, view_true=None, view_false=None, view_val=None,
               pre_proxy=False):
  # round eval values appropriately
  metric_names = {'coverage', 'density', 'precision', 'recall', 'train_acc', 'test_acc', 'fid'}
  header_list = [0, 1] if pre_proxy else [0, 1, 2]
  exp_table = pd.read_csv(log_file, index_col=0, header=header_list)
  rounding_vals = dict()
  for col in exp_table.columns:
    metric = col if not isinstance(col, tuple) else col[-1]
    if view_metrics is not None and metric in (metric_names - set(view_metrics)):
      exp_table.drop(columns=col, inplace=True)
    elif metric in {'coverage', 'density', 'precision', 'recall', 'train_acc', 'test_acc'}:
      rounding_vals[col] = 3
    elif metric in {'fid'}:
      rounding_vals[col] = 1
  # rounding_vals = {('mean', 'coverage'): 3, 'density': 3, 'precision': 3, 'recall': 3, 'fid': 1,
  #                  'train_acc': 3, 'test_acc': 3}
  exp_table = exp_table.round(rounding_vals)
  short_names = {'coverage': 'cover', 'density': 'densi', 'precision': 'preci', 'recall': 'recal'}
  exp_table = exp_table.rename(columns=short_names)

  for arg in view_true:
    exp_table = exp_table[exp_table[('args', arg)] == True]
  for arg in view_false:
    exp_table = exp_table[exp_table[('args', arg)] == False]
  for arg in view_val:
    arg, value = arg.split(':')
    exp_table = exp_table[exp_table[('args', arg)] == value]

  with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 0):
    print(exp_table)


def create_exp_tables(arg, log_dir):
  exp_table = experiment_table(log_dir)
  exp_table.to_csv(os.path.join(log_dir, 'results_all_args.csv'))
  reduced_exp_table = remove_matching_args(exp_table)
  reduced_exp_table = remove_redundant_args(reduced_exp_table, arg.redundant_cols)
  reduced_exp_table.to_csv(os.path.join(log_dir, 'results_run_args.csv'))
  final_exp_table = final_iter_table(reduced_exp_table)
  final_exp_table.to_csv(os.path.join(log_dir, 'results_final_it.csv'))
  best_exp_table = best_iter_table(reduced_exp_table)
  best_exp_table.to_csv(os.path.join(log_dir, 'best_results.csv'))

  avgn = arg.average_groups_of_n
  if avgn is not None:
    assert len(exp_table.index) % avgn == 0  # num ber of runs is a multiple of avgn
    assert all([int(k) == j for j, k in enumerate(exp_table.index)])  # all indices are present

    reduced_exp_table_avg_n = average_n_indices(reduced_exp_table, avgn)
    reduced_exp_table_avg_n.to_csv(os.path.join(log_dir, f'results_run_args_avg_{avgn}.csv'))
    final_exp_table_avg_n = final_iter_table(reduced_exp_table_avg_n)  #  .droplevel(level=0, axis=1)
    final_exp_table_avg_n.to_csv(os.path.join(log_dir, f'results_final_it_avg_{avgn}.csv'))
    best_exp_table_avg_n = best_iter_table(reduced_exp_table_avg_n, iter_idx=0)  #.droplevel(level=0, axis=1)
    best_exp_table_avg_n.to_csv(os.path.join(log_dir, f'best_results_avg_{avgn}.csv'))
    view_table(os.path.join(log_dir, f'best_results_avg_{avgn}.csv'), arg.view_metrics,
               arg.view_true, arg.view_false, arg.view_val, pre_proxy=arg.pre_proxy)
  else:
    best_path = os.path.join(log_dir, 'best_results.csv')
    if not os.path.exists(best_path):
      best_path = os.path.join(log_dir, 'results_final_it.csv')
    view_table(best_path, arg.view_metrics, arg.view_true, arg.view_false, arg.view_val,
               pre_proxy=True)


def get_best_steps(run_dir):
  ckpt = pt.load(os.path.join(run_dir, 'ckpt.pt'), map_location=pt.device('cpu'))
  best_fid_step = ckpt['best_step']
  best_proxy_step = ckpt['best_proxy_step']
  del ckpt
  return best_fid_step, best_proxy_step


def print_best_steps(exp_dir):
  run_dirs = get_rundirs(exp_dir)
  for dir_key in run_dirs:
    best_fid_step, best_proxy_step = get_best_steps(run_dirs[dir_key])
    print(f'run {dir_key}: best fid at {best_fid_step}, best proxy at {best_proxy_step}')


def main():
  parser = argparse.ArgumentParser()

  # PARAMS YOU LIKELY WANT TO SET
  parser.add_argument('--base_logdirs', default=['/home/fharder/dp-gfmn/logs/',
                                                 '/home/frederik/PycharmProjects/dp-gfmn/logs/'])
  parser.add_argument('--logdirs', '-l', nargs='*')
  parser.add_argument('--intermediate_results', '-ir', action='store_true')
  parser.add_argument('--average_groups_of_n', '-avgn', type=int, default=None)
  parser.add_argument('--redundant_cols', type=str, nargs='*', default=['log_dir', 'exp_name',
                                                                        'first_batch_id',
                                                                        'log_messages'])
  parser.add_argument('--view', type=str, default=None)
  parser.add_argument('--view_metrics', type=str, nargs='*', default=None)
  parser.add_argument('--view_true', type=str, nargs='*', default=[])
  parser.add_argument('--view_false', type=str, nargs='*', default=[])
  parser.add_argument('--view_val', type=str, nargs='*', default=[])

  parser.add_argument('--pre_proxy', action='store_true', help='set if old log without proxy fid')
  parser.add_argument('--print_best_steps', '-s', action='store_true', help='print best iterations')
  arg = parser.parse_args()
  base_logdir = None
  for base_dir in arg.base_logdirs:
    if os.path.exists(base_dir):
      base_logdir = base_dir
      break
  assert base_logdir
  for subdir in arg.logdirs:
    log_dir = os.path.join(base_logdir, subdir)
    if arg.view is not None:
      view_table(os.path.join(log_dir, arg.view), arg.view_metrics, arg.view_true, arg.view_false,
                 arg.view_val)
    elif has_any_results(log_dir):
      create_exp_tables(arg, log_dir)
    if arg.print_best_steps:
      print_best_steps(log_dir)


if __name__ == '__main__':
  main()
