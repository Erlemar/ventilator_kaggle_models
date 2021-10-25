import argparse
import glob
import os

import numpy as np
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference in Ventilator competition')
    parser.add_argument('--run_name', help='folder_name', type=str, default='2021-10-23_17-00-34')
    parser.add_argument('--averaging', help='averaging method', type=str, default='median')
    args = parser.parse_args()

    sub_paths = glob.glob(f'outputs/{args.run_name}/*/submission*.csv')
    oof_paths = glob.glob(f'outputs/{args.run_name}/*/*oof*.csv')



    for p in sub_paths:
        assert os.path.exists(p), f'No {p = }'
        print(os.path.exists(p), p)

    for p in oof_paths:
        assert os.path.exists(p), f'No {p = }'
        print(os.path.exists(p), p)

    assert len(sub_paths) == 20, 'Not 20 files!'
    assert len(oof_paths) == 20, 'Not 20 files!'


    sub1 = pd.read_csv(sub_paths[0])
    predictions = np.zeros((len(sub1), len(sub_paths)))
    predictions[:, 0] = sub1['pressure'].values
    scores = []
    # folder_path = '/'.join(sub_paths[0].split('/')[:-1])
    # name = [i for i in os.listdir(f'{folder_path}/saved_models') if 'best' in i][0]
    scores.append(float(sub_paths[0][-10:-4]))

    for i, file_name in enumerate(sub_paths[1:]):
        print(file_name)
        sub = pd.read_csv(file_name)
        predictions[:, i + 1] = sub['pressure']
        folder_path = '/'.join(file_name.split('/')[:-1])
        # name = [i for i in os.listdir(f'{folder_path}/saved_models') if 'best' in i][0]
        scores.append(float(file_name[-10:-4]))

    sub1['pressure'] = np.mean(predictions, 1)
    mean_path = f'outputs/sub_20_folds_mean_{args.run_name}.csv'
    print(mean_path)
    sub1.to_csv(mean_path, index=False)

    sub1['pressure'] = np.median(predictions, 1)
    median_path = f'outputs/sub_20_folds_median_{args.run_name}.csv'
    print(median_path)
    sub1.to_csv(median_path, index=False)

    print('scores', scores)
    print(f'{np.mean(scores) = }')
    print(f'{np.median(scores) = }')

    # train = pd.read_csv('d:/DataScience/Python_projects/Current_projects/GBVPP/data/train.csv')
    train = pd.read_csv('/workspace/data/ventilator_pressure_prediction/train.csv')
    all_pressure = sorted(train['pressure'].unique())
    pressure_min = all_pressure[0]
    pressure_max = all_pressure[-1]
    pressure_step = all_pressure[1] - all_pressure[0]
    print(pressure_min, pressure_max, pressure_step)
    sub1['pressure'] = np.round((sub1.pressure - pressure_min) / pressure_step) * pressure_step + pressure_min
    median_pp_path = f'outputs/sub_20_folds_median_pp_{args.run_name}.csv'
    print(median_pp_path)
    sub1.to_csv(median_pp_path, index=False)

    sub1 = pd.read_csv(oof_paths[0])

    for i, file_name in enumerate(oof_paths[1:]):
        sub = pd.read_csv(file_name)
        sub1['pressure'] += sub['pressure']

    oof_path = f'outputs/oof_20_folds_{args.run_name}.csv'
    print(oof_path)
    sub1.to_csv(oof_path, index=False)

