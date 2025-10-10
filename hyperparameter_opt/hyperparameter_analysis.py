import json
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

import os

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

# Helper methods
def extract_df(trial: str, keep: list[str] = None) -> pd.DataFrame:
    path = f'{trial}/progress.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.dropna(axis=0, inplace=True)
        if keep is not None:
            drop = [c for c in df.columns if c not in keep]
            df.drop(labels=drop, axis=1, inplace=True)
        return df
    return None


def extract_all_df(experiment: str, threshold: int = 0) -> pd.DataFrame:
    trials = [x for x in os.listdir(experiment) if os.path.isdir(f'{experiment}/{x}')]
    df = pd.DataFrame([])
    for trial in trials:
        new = extract_df(f'{experiment}/{trial}')
        if (new is not None):
            new['trial'] = trial
            if len(new) > threshold:
                df = pd.concat((df, new.iloc[threshold:, :]), ignore_index=True)
    df.reset_index(inplace=True, drop=True)
    return df

def identify_best(
    experiment: str,
    target_metric: str,
    threshold: int = 0,
) -> pd.DataFrame:
    df = extract_all_df(experiment, threshold)
    g1: DataFrameGroupBy = df.groupby(by='trial')
    best_iter = g1.apply(lambda g: g.loc[g[target_metric].idxmax()]['training_iteration'], include_groups=False)
    best_iter.name = 'Best_Iter'
    df1 = g1.aggregate({target_metric: 'max', 'training_iteration': 'max'})
    df1.rename(columns={target_metric: 'Metric', 'training_iteration': 'Iters'}, inplace=True)
    df1.sort_values(by='Metric', ascending=False, inplace=True)
    df1 = df1.join(other=best_iter)
    return df1

# Core methods
def create_readable_csv(exp_dir):
    col_map = {
        'num_env_steps_sampled_lifetime': 'EnvSteps',
        'num_episodes_lifetime':'Episodes',
        'training_iteration': 'Iters',
        'time_this_iter_s': 'TimeThisIter',
        'time_total_s': 'TimeTotal',
        'env_runners/episode_len_mean': 'EpisodeLengthMean',
        'env_runners/episode_return_mean': 'EpisodeReturnMean',
    }
    to_keep = list(col_map.keys())
    for trial in os.listdir(exp_dir):
        trial_dir = f'{exp_dir}/{trial}'
        if os.path.isdir(trial_dir):
            df = extract_df(trial_dir, to_keep)
            if (df is not None):
                df.rename(columns=col_map, inplace=True)
                df = df.round(4)
                df.to_csv(f'hyperparameter_opt/analysis/{trial}.csv', index=False)

def hyperparameter_extraction(exp_dir) -> pd.DataFrame:
    info = []
    for trial in os.listdir(exp_dir):
        trial_dir = f'{exp_dir}/{trial}'
        if os.path.isdir(trial_dir) and len(list(os.listdir(trial_dir))) > 0:
            with open(f'{trial_dir}/params.json', 'r') as file:
                data = json.load(file)
            info.append({
                'trial': trial,
                # General hyperparameters
                'minibatch_size': data['minibatch_size'],
                'lr': data['lr'],
                'gamma': data['gamma'],
                'lambda_': data['lambda'],
                'clip_param': data['clip_param'],
                'grad_clip': data['grad_clip'],
                # KL hyperparameters
                'kl_target': data['kl_target'],
                'kl_coeff': data['kl_coeff'],
                'use_kl_loss': data['use_kl_loss'],
                # Value function hyperparameters
                'vf_clip_param': data['vf_clip_param'],
                'vf_loss_coeff': data['vf_loss_coeff'],
                'vf_share_layers': data['vf_share_layers'],
            })
    info = pd.DataFrame(info)
    info.set_index('trial', inplace=True)
    return info

def find_best(exp_dir):
    best = identify_best(
        experiment=exp_dir,
        target_metric="env_runners/episode_return_mean"
    )
    hyper = hyperparameter_extraction(exp_dir)
    info = pd.merge(best, hyper, left_index=True, right_index=True)
    info.to_csv('hyperparameter_opt/analysis/trial_info.csv')

# python hyperparameter_analysis.py --experiment-name "Test_Hyperparameter_Search"
parser = argparse.ArgumentParser()
parser.add_argument("--experiment-name", type=str)
parser.add_argument("--create-plots", type=str, default=None)
args = parser.parse_args()
exp_dir = f'hyperparameter_opt/results/{args.experiment_name}'
create_readable_csv(exp_dir)
find_best(exp_dir)
