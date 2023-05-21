import os
import numpy as np
import argparse


def get_log_values(run_dirs, final_log_file, show_intermediate_log, intermediate_log_prefix):
  out_vals = []
  for run in run_dirs:
    final_log_path = os.path.join(run.path, final_log_file)

    if show_intermediate_log:
      int_fids = [f for f in os.scandir(run.path) if f.name.startswith(intermediate_log_prefix)]
      int_fids = [f for f in int_fids if f.name != final_log_file]
      # print([f.name for f in int_fids])
      int_fid_it = [int(f.name[len(intermediate_log_prefix):-len('.npy')]) for f in int_fids]
      int_fid_scores = [np.load(f.path) for f in int_fids]
      scores_by_it = sorted(zip(int_fid_it, int_fid_scores), key=lambda k: k[0])
    else:
      scores_by_it = None

    if os.path.exists(final_log_path):
      fid_score = np.load(final_log_path)
    else:
      fid_score = None
    out_vals.append((run.name, fid_score, scores_by_it))
  return out_vals


def check_matching_steps(out_vals, show_intermediate_log):
  matching_steps = False  # check if all runs have the same log steps
  if show_intermediate_log:
    matching_steps = True
    base_steps_reference = out_vals[0][2]
    for out_val in out_vals:
      if len(out_val[2]) > len(base_steps_reference):
        base_steps_reference = out_val[2]
    base_steps = [k[0] for k in base_steps_reference]
    for run_vals in out_vals:
      steps = [k[0] for k in run_vals[2]]
      if not all([i == j for i, j in zip(base_steps, steps)]):
        matching_steps = False
        break
  else:
    base_steps = None
  return matching_steps, base_steps


def aggregate_groups_of_runs(out_vals, aggregation_mode, aggegate_n, matching_steps):
  assert aggregation_mode in {'mean', 'std'}
  aggregation_fun = np.mean if aggregation_mode == 'mean' else np.std
  new_out_vals = []
  for idx, run_vals in enumerate(out_vals):
    if idx % aggegate_n == 0:
      new_name = out_vals[idx][0] + ' - ' + out_vals[idx + aggegate_n - 1][0]
      acc_vals = [out_vals[idx + k][1] for k in range(aggegate_n)]
      if any([k is None for k in acc_vals]):
        new_fid = np.nan
      else:
        new_fid = aggregation_fun(acc_vals)
      new_scores_by_it = []
      if out_vals[0][2] is not None:
        assert matching_steps
        for jdx in range(len(out_vals[idx][2])):
          acc_vals = [out_vals[idx + k][2][jdx][1] for k in range(aggegate_n)]
          if any([k is None for k in acc_vals]):
            avg_score = np.nan
          else:
            avg_score = aggregation_fun(acc_vals)
          score_it = out_vals[idx][2][jdx][0]
          new_scores_by_it.append((score_it, avg_score))
      new_out_vals.append((new_name, new_fid, new_scores_by_it))
  return new_out_vals


def f_format(maybe_float, print_res):
  if isinstance(maybe_float, (np.float32, np.float64, np.ndarray)):
    return f' {maybe_float:{print_res}f} '
  else:
    return ' ' + '-' * 6 + ' '


def print_log_results(out_vals, base_steps, matching_steps, metric_str, print_res):
  if matching_steps:
    print(f'| run id | final {metric_str} |' +
          '|'.join([f' {metric_str}@{k} ' for k in base_steps]) + '|')
    print('| ------ | --------- |' + ' --- |' * len(base_steps))

  for run_vals in out_vals:

    if run_vals[2] is None or len(run_vals[2]) == 0:
      intermediate_str = '|'.join([' --- '] * len(base_steps)) if base_steps is not None else ''
    else:
      filler_str = '|'.join([' --- '] * (len(base_steps) - len(run_vals[2])))
      if not matching_steps:
        intermediate_str = '|'.join([f' it{k[0]}: {k[1]:{print_res}f}' for k in run_vals[2]]) + '|' + \
                           filler_str
      else:
        intermediate_str = '|'.join([f_format(k[1], print_res) for k in run_vals[2]]) + '|' + filler_str

    out_str = f'| {run_vals[0]} |' + f_format(run_vals[1], print_res) + '|' + intermediate_str + '|'
    print(out_str)


def print_final_mean_and_std(mean_key_out_vals, std_key_out_vals, metric_str, print_res):
  print(f'| run ids | {metric_str} mean | {metric_str} std |')
  print('| ------ | --------- | ------- |')

  for mean_vals, std_vals in zip(mean_key_out_vals, std_key_out_vals):

    # intermediate_str = '|'.join([f' it{k[0]}: {k[1]:{print_res}f}' for k in run_vals[2]]) + '|' + \
    #                    filler_str
    # else:
    #   intermediate_str = '|'.join([f_format(k[1], print_res) for k in run_vals[2]]) + '|' + filler_str

    out_str = f'| {mean_vals[0]} | {f_format(mean_vals[1], print_res)} | {f_format(std_vals[1], print_res)} |'
    print(out_str)

def get_rundirs(log_dir):
  run_dirs = [d for d in os.scandir(log_dir) if (d.is_dir() and d.name.startswith('run'))]
  if all([d.name.startswith('run_') for d in run_dirs]):
    return sorted(run_dirs, key=lambda d: int(d.name[len('run_'):]))
  else:
    return sorted(run_dirs, key=lambda d: d.name)


def gather_fid_scores(log_dir, show_intermediate_fid, average_groups_of_n, standard_dev_groups_of_n,
                      fid_file='fid.npy', intermediate_fid_prefix='fid_it'):
  run_dirs = get_rundirs(log_dir)
  print(f'fid scores for {log_dir}:')

  out_vals = get_log_values(run_dirs, fid_file, show_intermediate_fid, intermediate_fid_prefix)
  matching_steps, base_steps = check_matching_steps(out_vals, show_intermediate_fid)
  mean_out_vals, std_out_vals = None, None
  if average_groups_of_n is not None:
    mean_out_vals = aggregate_groups_of_runs(out_vals, 'mean', average_groups_of_n, matching_steps)

  if standard_dev_groups_of_n is not None:
    std_out_vals = aggregate_groups_of_runs(out_vals, 'std', standard_dev_groups_of_n, matching_steps)

  if mean_out_vals and std_out_vals:
    print_final_mean_and_std(mean_out_vals, std_out_vals, metric_str='FID', print_res=6.2)
  else:
    if mean_out_vals or std_out_vals:
      out_vals = mean_out_vals if mean_out_vals else std_out_vals
    print_log_results(out_vals, base_steps, matching_steps, metric_str='FID', print_res=6.2)


def gather_accuracies(log_dir, show_intermediate_fid, average_groups_of_n, standard_dev_groups_of_n,
                      accuracy_file='accuracies.npz', intermediate_accuracy_prefix='accuracies_it'):
  run_dirs = get_rundirs(log_dir)
  print(f'accuracy scores for {log_dir}:')
  out_vals = get_log_values(run_dirs, accuracy_file, show_intermediate_fid,
                            intermediate_accuracy_prefix)
  accuracy_keys = ['mean_acc', 'logistic_reg', 'mlp', 'xgboost',
                   'train_acc', 'test_acc']
  for acc_key in accuracy_keys:
    if acc_key in out_vals[0][1].keys():
      print(f'results for {acc_key}')
      key_out_vals = []
      for val in out_vals:
        intermediate_val_select = [(i, k[acc_key]) for i, k in val[2]] if val[2] is not None else None
        final_val = val[1][acc_key] if val[1] is not None else None
        key_out_vals.append((val[0], final_val, intermediate_val_select))
      matching_steps, base_steps = check_matching_steps(key_out_vals, show_intermediate_fid)
      mean_key_out_vals, std_key_out_vals = None, None
      if average_groups_of_n is not None:
        mean_key_out_vals = aggregate_groups_of_runs(key_out_vals, 'mean', average_groups_of_n,
                                                     matching_steps)
      if standard_dev_groups_of_n is not None:
        std_key_out_vals = aggregate_groups_of_runs(key_out_vals, 'std', standard_dev_groups_of_n,
                                                    matching_steps)
        # key_out_vals = average_groups_of_runs(key_out_vals, average_groups_of_n, matching_steps)
      if mean_key_out_vals and std_key_out_vals:
        print_final_mean_and_std(mean_key_out_vals, std_key_out_vals, metric_str='ACC', print_res=6.3)
      else:
        if mean_key_out_vals or std_key_out_vals:
          key_out_vals = mean_key_out_vals if mean_key_out_vals else std_key_out_vals
        print_log_results(key_out_vals, base_steps, matching_steps, metric_str='ACC', print_res=6.3)


def has_certain_log_files(log_dir, desired_prefix):
  run_dirs = [d for d in os.scandir(log_dir) if (d.is_dir() and d.name.startswith('run'))]
  for run in run_dirs:
    int_fids = [f for f in os.scandir(run.path) if f.name.startswith(desired_prefix)]
    if int_fids:
      return True
  return False


def main():
  parser = argparse.ArgumentParser()

  # PARAMS YOU LIKELY WANT TO SET
  parser.add_argument('--base_logdirs', default=['/home/fharder/dp-gfmn/logs/',
                                                 '/home/frederik/PycharmProjects/dp-gfmn/logs/'])
  parser.add_argument('--logdirs', '-l', nargs='*')
  parser.add_argument('--intermediate_results', '-ir', action='store_true')
  parser.add_argument('--average_groups_of_n', '-avgn', type=int, default=None)
  parser.add_argument('--standard_dev_groups_of_n', '-stdn', type=int, default=None)

  arg = parser.parse_args()
  assert not (arg.average_groups_of_n and arg.standard_dev_groups_of_n and arg.intermediate_results)
  base_logdir = None
  for base_dir in arg.base_logdirs:
    if os.path.exists(base_dir):
      base_logdir = base_dir
      break
  assert base_logdir
  for subdir in arg.logdirs:
    log_dir = os.path.join(base_logdir, subdir)
    if has_certain_log_files(log_dir, desired_prefix='fid'):
      gather_fid_scores(log_dir, arg.intermediate_results,
                        arg.average_groups_of_n, arg.standard_dev_groups_of_n)
    if has_certain_log_files(log_dir, desired_prefix='accuracies'):
      gather_accuracies(log_dir, arg.intermediate_results,
                        arg.average_groups_of_n, arg.standard_dev_groups_of_n)


if __name__ == '__main__':
  main()
