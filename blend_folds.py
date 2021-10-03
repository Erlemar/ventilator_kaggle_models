import argparse
import glob

import numpy as np
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference in Ventilator competition')
    parser.add_argument('--run_name', help='folder_name', type=str, default='2021-09-25_16-58-39')
    parser.add_argument('--averaging', help='averaging method', type=str, default='median')
    args = parser.parse_args()

    file_names = glob.glob(f'outputs/{args.run_name}/*/*.csv')
    sub1 = pd.read_csv(file_names[0])

    if args.averaging == 'mean':
        for file_name in file_names[1:]:
            sub = pd.read_csv(file_name)
            sub1['pressure'] += sub['pressure']
        sub1['pressure'] /= len(file_names)

    elif args.averaging == 'median':
        predictions = np.zeros((len(sub1), 5))
        predictions[:, 0] = sub1['pressure'].values
        for i, file_name in enumerate(file_names[1:]):
            sub = pd.read_csv(file_name)
            predictions[:, i + 1] = sub['pressure']
        sub1['pressure'] = np.median(predictions, 1)

    else:
        print('Wrong averaging type')

    sub1.to_csv(f'outputs/{args.run_name}_blend.csv', index=False)