import copy
import datetime
import json
import logging
import os
import sys
import time
import torch
import numpy as np
from utils import factory
# from utils.data_manager_attribution import DataManager
# from utils.data_manager_disentangle import DataManager
from utils.data_manager import DataManager
from utils.toolkit import ConfigEncoder, count_parameters, load_model, set_random


def eval(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _eval(args)


def _eval(args):
    set_random(args['seed'])
    time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
    args['time_str'] = time_str
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    exp_name = "{}_{}_{}_{}_{}_B{}_Inc{}".format(
        args["time_str"],
        args["dataset"],
        args["split"],
        args["convnet_type"],
        args["seed"],
        init_cls,
        args["increment"],
    )
    args['exp_name'] = exp_name

    if args['debug']:
        logfilename = "logs/debug/{}/{}/{}/{}/{}".format(
            args["prefix"],
            args["dataset"],
            args['split'],
            args["model_name"],
            args["exp_name"]
        )
    else:
        logfilename = "logs/{}/{}/{}/{}/{}".format(
            args["prefix"],
            args["dataset"],
            args["split"],
            args["model_name"],
            args["exp_name"]
        )

    args['logfilename'] = logfilename

    csv_name = "{}_{}_{}_{}_B{}_Inc{}".format(
        args["dataset"],
        args["split"],
        args["seed"],
        args["convnet_type"],
        init_cls,
        args["increment"],
    )
    args['csv_name'] = csv_name
    if args['save_name']:
        os.makedirs(args['save_name'], exist_ok=True)
        log_path = os.path.join(args['save_name'], "main.log")
        config_filepath = os.path.join(args['save_name'], 'configs.json')
    else:
        os.makedirs(logfilename, exist_ok=True)
        log_path = os.path.join(args["logfilename"], "main.log")
        config_filepath = os.path.join(args["logfilename"], 'configs.json')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(f"Time Str >>> {args['time_str']}")
    # save config
    with open(config_filepath, "w") as fd:
        json.dump(args, fd, indent=2, sort_keys=True, cls=ConfigEncoder)

    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["train_data_path"],
        args["val_data_path"],
        args["test_data_path"],
        args["out_data_path"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args["debug"]
    )

    model = factory.get_model(args)
    logging.info("All params: {}".format(count_parameters(model._network)))
    logging.info(
        "Trainable params: {}".format(count_parameters(model._network, True))
    )

    start_time = time.time()
    logging.info(f"Start time:{start_time}")
    for task in range(1):
        set_random(args['seed'])
        model.incremental_eval(data_manager)

    end_time = time.time()
    logging.info(f"End Time:{end_time}")


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus



def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def save_time(args, cost_time):
    _log_dir = os.path.join("./results/", "times", f"{args['prefix']}")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    with open(_log_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']}, {cost_time} \n")


def save_results(args, cnn_accy, nme_accy, eval_type):
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top1")
    os.makedirs(_log_dir, exist_ok=True)

    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in list(cnn_accy['grouped'].values())[:-4]:
                f.write(f"{_acc},")
            f.write(f"{list(cnn_accy['grouped'].values())[-3]}\n")
    else:
        assert args['prefix'] in ['fair', 'auc']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in list(cnn_accy['grouped'].values())[:-4]:
                f.write(f"{_acc},")
            f.write(f"{list(cnn_accy['grouped'].values())[-3]}\n")

    if eval_type == 'NME':
        _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top1")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
        if args['prefix'] == 'benchmark':
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},")
                for _acc in list(nme_accy['grouped'].values())[:-4]:
                    f.write(f"{_acc},")
                f.write(f"{list(nme_accy['grouped'].values())[-3]}\n")
        else:
            assert args['prefix'] in ['fair', 'auc']
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
                for _acc in list(nme_accy['grouped'].values())[:-4]:
                    f.write(f"{_acc},")
                f.write(f"{list(nme_accy['grouped'].values())[-3]}\n")


def eval_open(_out_k, _labels_k, model, eval_type, args, final=False):
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "open-set")
    os.makedirs(_log_dir, exist_ok=True)
    out_auc, out_oscr = model.eval_out_task(_out_k, _labels_k, type='out')
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark' and final:
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']}, {args['model_name']},")
            f.write(f"{out_auc}, {out_oscr}\n")
    logging.info("open-set-out {} auc: {}, oscr: {}".format(eval_type, out_auc, out_oscr))

