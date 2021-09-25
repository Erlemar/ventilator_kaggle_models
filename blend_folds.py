import argparse
import glob

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference in Ventilator competition')
    parser.add_argument('--run_name', help='folder_name', type=str, default='2021-09-25_16-58-39')
    args = parser.parse_args()

    file_names = glob.glob(f'outputs/{args.run_name}/*/*.csv')
    sub1 = pd.read_csv(file_names[0])
    for file_name in file_names[1:]:
        sub = pd.read_csv(file_name)
        sub1['pressure'] += sub['pressure']
    sub1['pressure'] /= len(file_names)
    sub1.to_csv(f'outputs/{args.run_name}_blend.csv', index=False)
