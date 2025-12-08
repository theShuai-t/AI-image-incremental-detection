import json
import argparse
import os

import yaml

from trainer import train
import evaler
import trainer

def main():
    args = setup_parser().parse_args()
    args.config = f"./configs/{args.model_name}.json"
    args.yaml_file = f"./configs/data_list.yaml"
    param = load_json(args.config)
    data_list = load_yaml(args.yaml_file, args.split)
    args = vars(args)  # Converting argparse Namespace to a dict.
    param.update(args)
    param.update(data_list)
    if args['run_type'] == 'train':
        trainer.train(param)
    elif args['run_type'] == 'test':
        evaler.eval(param)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def load_yaml(yaml_path, split):
    with open(yaml_path) as file:
        data_list = yaml.load(file, Loader=yaml.FullLoader)[split]
    return data_list


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')

    parser.add_argument('--run_type', type=str, default='train')
    parser.add_argument('--dataset', type=str, default="OSMA")
    parser.add_argument('--split', type=str, default='split1')
    parser.add_argument('--memory_size', '-ms', type=int, default=100)
    parser.add_argument('--init_cls', '-init', type=int, default=2)
    parser.add_argument('--increment', '-incre', type=int, default=1)
    parser.add_argument('--model_name', '-model', type=str, default=None, required=True)
    parser.add_argument('--convnet_type', '-net', type=str, default='resnet32')
    parser.add_argument('--prefix', '-p', type=str, help='exp type', default='benchmark',
                        choices=['benchmark', 'fair', 'auc'])
    parser.add_argument('--device', '-d', nargs='+', type=int, default=[0])
    parser.add_argument('--choose_idx', '-idx', type=int, default=1)
    parser.add_argument('--save_name', '-name', type=str, default=None)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--skip', action="store_true")
    parser.add_argument('--param', type=float, default=0)
    parser.add_argument('--shot', type=int, default=100)

    return parser


if __name__ == '__main__':
    main()
