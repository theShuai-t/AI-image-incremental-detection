import logging
import os
import random
import shutil
import numpy as np
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score


class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


def read_annotations(data_path, debug=False):
    # str.strip() 去掉字符串首尾的空格、\n、\t等
    lines = map(str.strip, open(data_path).readlines())
    data = []
    targets = []
    for line in lines:
        sample_path, target = line.split('\t')
        target = int(target)
        data.append(sample_path)
        targets.append(target)
    # random.shuffle(data)
    if debug:
        data = data[:1000]
        targets = targets[:1000]
    return np.array(data), np.array(targets)




def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def metrics_others(y_order, y_pred, y_true, increment=1):
    all_precision_fake, all_precision_real, all_recall_fake, all_recall_real, all_f1_fake, all_f1_real = {}, {}, {}, {}, {}, {}
    # all_precision["total"] = np.around(precision_score(y_true, y_pred) * 100, decimals=2)
    # all_recall["total"] = np.around(recall_score(y_true, y_pred) * 100, decimals=2)
    # all_f1["total"] = np.around(f1_score(y_true, y_pred) * 100, decimals=2)
    # Grouped accuracy
    for class_id in range(0, np.max(y_order) + 1, increment):
        label = "{}".format(
            str(class_id).rjust(2, "0")
        )
        idxes = np.where(y_order == class_id)
        if len(idxes[0]) == 0:
            all_precision_fake[label] = 'nan'
            all_precision_real[label] = 'nan'
            all_recall_fake[label] = 'nan'
            all_recall_real[label] = 'nan'
            all_f1_fake[label] = 'nan'
            all_f1_real[label] = 'nan'
            continue
        all_precision_fake[label] = np.around(precision_score(y_true[idxes], y_pred[idxes]) * 100, decimals=2)
        all_precision_real[label] = np.around(precision_score(y_true[idxes], y_pred[idxes], pos_label=0) * 100, decimals=2)
        all_recall_fake[label] = np.around(recall_score(y_true[idxes], y_pred[idxes]) * 100, decimals=2)
        all_recall_real[label] = np.around(recall_score(y_true[idxes], y_pred[idxes], pos_label=0) * 100, decimals=2)
        all_f1_fake[label] = np.around(f1_score(y_true[idxes], y_pred[idxes]) * 100, decimals=2)
        all_f1_real[label] = np.around(f1_score(y_true[idxes], y_pred[idxes], pos_label=0) * 100, decimals=2)

    return all_precision_fake, all_precision_real, all_recall_fake, all_recall_real, all_f1_fake, all_f1_real

def accuracy(y_order, y_pred, y_true, nb_old, increment=1):
    all_acc = {}
    all_acc["total"] = np.around(
        np.sum(y_pred == y_true) * 100 / len(y_true), decimals=2
    )
    # Grouped accuracy
    for class_id in range(0, np.max(y_order) + 1, increment):
        label = "{}".format(
            str(class_id).rjust(2, "0")
        )
        idxes = np.where(y_order == class_id)
        if len(idxes[0]) == 0:
            all_acc[label] = 'nan'
            continue
        all_acc[label] = np.around(
            np.sum(y_pred[idxes] == y_true[idxes]) * 100 / len(idxes[0]), decimals=2
        )
        # real = np.size(np.where((y_pred[idxes] == y_true[idxes]) & (y_true[idxes] == np.zeros_like(y_true[idxes]))))
        # real_total = np.sum(y_true[idxes] == np.zeros_like(y_true[idxes]))
        # fake = np.size(np.where((y_pred[idxes] == y_true[idxes]) & (y_true[idxes] == np.ones_like(y_true[idxes]))))
        # fake_total = np.sum(y_true[idxes] == np.ones_like(y_true[idxes]))
        # logging.info("label: {}, real: {}/{}, fake: {}/{}".format(label, real, real_total, fake, fake_total))

    # Old accuracy

    # idxes = np.where(y_order < nb_old)[0]
    # all_acc["old"] = (
    #     0
    #     if len(idxes) == 0
    #     else np.around(
    #         np.sum(y_pred[idxes] == y_true[idxes]) * 100 / len(idxes), decimals=2
    #     )
    # )
    #
    # # New accuracy
    # idxes = np.where(y_order >= nb_old)[0]
    # all_acc["new"] = np.around(
    #     np.sum(y_pred[idxes] == y_true[idxes]) * 100 / len(idxes), decimals=2
    # )

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


def save_fc(args, best_acc, cur_task, model):
    _path = os.path.join(args['logfilename'], "task" + str(cur_task), "val_acc" + str(best_acc) + "_fc.pth")
    # if len(args['device']) > 1:
    #     fc_weight = model._network.fc.weight.data
    # else:
    #     fc_weight = model._network.fc.weight.data.cpu()
    fc_weight = model.fc.weight.data
    torch.save(fc_weight, _path)


def save_model(args, best_acc, cur_task, best_model_path, model, skip_checkpoints=False, out_acc=None, centers=None,
               radius=None):
    if skip_checkpoints:
        root_path = "checkpoints"
        os.makedirs(root_path, exist_ok=True)
        checkpoints_path = os.path.join(root_path, f"{args['model_name']}_{args['csv_name']}_0.pth")
        old_path = best_model_path[-1]
        shutil.copyfile(old_path, checkpoints_path)
        logging.info(
            'save checkpoints successfully! old path: {}, checkpoints path: {}'.format(old_path, checkpoints_path))
        return checkpoints_path

    # used in PODNet
    _path = os.path.join(args['logfilename'], "task" + str(cur_task))
    os.makedirs(_path, exist_ok=True)
    if out_acc:
        _path = os.path.join(_path, "val_acc" + str(best_acc) + "test_acc" + str(out_acc) + "_model.pth")
    else:
        _path = os.path.join(_path, "val_acc" + str(best_acc) + "_model.pth")
    if centers is not None:
        state = {
            'network': model.state_dict(),
            'centers': centers,
            'radius': radius,
        }
        torch.save(state, _path)
    else:
        torch.save(model.state_dict(), _path)
    return _path


def load_model(model, args, best_model_path, trained_path=None, have_centers=False):
    # trained_path 用于stitch pretrain 用于训练的预训练权重
    state = None
    if args["skip"] and len(best_model_path) == 0:
        # path = f"checkpoints/finetune_{args['csv_name']}_0.pth"
        # path = f"/home/tangshuai/domain/logs/benchmark/OSMA/split4/vit_xception_center/without R&tri init 1e-3/task0/val_acc99.9_model.pth"
        # path = f"/home/tangshuai/domain/logs/benchmark/OSMA/split4/vit_xception_center/current best without R 0.1kd 0.01tri 2e-3/task2/val_acc96.33_model.pth"
        # path = f"/data/hdd1/tangshuai/domain/logs/benchmark/OSMA/split2/split2/vit_xception_center/final_all_block/task0/val_acc99.45_model.pth"
        # path = '/data/hdd1/tangshuai/domain/logs/benchmark/OSMA/OSMA/split5/vit_xception_center/suspend/task0/val_acc100.0_model.pth'
        path = '/home/tangshuai/domain/logs/benchmark/OSMA/split2/vit_xception_center/ours_consistent/task0/val_acc99.65_model.pth'
        # path = '/home/tangshuai/domain/logs/benchmark/OSMA/split5/vit_xception_center/ours_consistent/task0/val_acc100.0_model.pth'
        # path = f"/home/tangshuai/domain/logs/benchmark/OSMA/split2/vit_xception_center/ablation archi xception adapter/task0/val_acc99.45_model.pth"
        # path = f"/home/tangshuai/domain/logs/benchmark/OSMA/split2/vit_xception_center/ablation archi xception adapter/task0/val_acc99.65_model.pth"

        # split9
        # path = "/data/hdd1/tangshuai/domain/logs/benchmark/OSMA/OSMA/split4/vit_swin/suspend/task0/val_acc100.0_model.pth" # dynamicSwin
        # path = "/data/hdd1/tangshuai/domain/logs/benchmark/OSMA/OSMA/split4/vit_xception/icarl/task0/val_acc99.9_model.pth" # icarl
        # path = "/data/hdd1/tangshuai/domain/logs/benchmark/OSMA/OSMA/split4/vit_xception_center/suspend/task0/val_acc100.0_model.pth" # ours
        # path = "/data/hdd1/tangshuai/domain/logs/benchmark/OSMA/OSMA/split4/npr/npr/task0/val_acc100.0_model.pth" # npr
        # path = "/data/hdd1/tangshuai/domain/logs/benchmark/OSMA/OSMA/split4/universe/universe/task0/val_acc100.0_model.pth" # universe
        # path = "/data/hdd1/tangshuai/domain/logs/benchmark/OSMA/OSMA/split5/sprompts/sprompts/task0/val_acc99.3_model.pth" # Sprompts
        # path = "/home/tangshuai/domain/logs/benchmark/OSMA/split9/sprompts/sprompts sinet 5/task1/val_acc98.9_model.pth" # Sprompts

        # path = "/data/hdd1/tangshuai/domain/logs/benchmark/OSMA/OSMA/split5/rawm/rawm/task0/val_acc99.8_model.pth" # RAWM
        # path = "/data/hdd1/tangshuai/domain/logs/benchmark/OSMA/OSMA/split5/dfil_xception/dfil/task0/val_acc99.7_model.pth" # DFIL

        # seed
        # path = "/home/tangshuai/domain/logs/benchmark/OSMA/split2/vit_xception_center/suspend seed 0/task0/val_acc99.65_model.pth" # seed 0
        # path = "/home/tangshuai/domain/logs/benchmark/OSMA/split2/vit_xception_center/suspend seed 1/task0/val_acc99.75_model.pth" # seed 1


        state = torch.load(path, map_location=args["device"][0])
        model.load_state_dict(state, strict=False)
        # path = f"checkpoints/vit_{args['csv_name']}_0.pth"
        # path = f"checkpoints/rawm_{args['csv_name']}_0.pth"
        # path = f"checkpoints/disentangle_{args['csv_name']}_0.pth"
    else:
        path = best_model_path[-1]
        state = torch.load(path, map_location=args["device"][0])
        model.load_state_dict(state)
    if trained_path:
        path = trained_path
        if have_centers:
            state = torch.load(path, map_location=args["device"][0])
            model.load_state_dict(state['network'])
        else:
            state = torch.load(path, map_location=args["device"][0])
            model.load_state_dict(state)
        logging.info("load best model successfully! best model: {}".format(path))
    logging.info("load best model successfully! best model: {}".format(path))
    return state


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # CUDA >= 10.2版本会提示设置这个环境变量
    torch.use_deterministic_algorithms(True)


def early_stop(args, best_acc, val_acc, patience, best_model_path, cur_task, model, centers=None, radius=None):
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_path.append(
            save_model(args, best_acc, cur_task, best_model_path, model, centers=centers, radius=radius))
        # save_fc(args, best_acc, cur_task, model)
        if len(best_model_path) > 3:
            os.remove(best_model_path[0])
            # os.remove(best_model_path[0].replace('model', 'fc'))
            best_model_path.pop(0)
        patience = 0
    else:
        patience += 1
    return best_acc, patience


# RKD
def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss
