import copy
import logging
from time import sleep

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy, set_random
from utils.draw import plot_confusion
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

EPSILON = 1e-8
batch_size = 100
num_workers = 8


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._real_data_memory, self._fake_data_memory = np.array([]), np.array([])
        self._real_targets_memory, self._fake_targets_memory = np.array([]), np.array([])

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        data_length = len(np.concatenate((self._real_data_memory, self._fake_data_memory)))
        targets_length = len(np.concatenate((self._real_targets_memory, self._fake_targets_memory)))
        assert data_length == targets_length, "Exemplar size error."
        return data_length

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        self._reduce_exemplar(data_manager, per_class)
        self._construct_exemplar(data_manager, per_class)

    def after_task(self):
        pass

    def _evaluate(self, y_order, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_order, y_pred, y_true, self._known_classes, self.args['increment'])
        ret["grouped"] = grouped
        return ret

    def eval_task(self, type='test'):
        if type == 'out':
            loader = self.out_loader
        else:
            loader = self.test_loader

        y_order, y_out, y_true = self._eval_cnn(type, loader)
        y_pred = np.argmax(y_out, axis=1)
        cnn_accy = self._evaluate(y_order, y_pred, y_true)

        if 'out' in type:
            return y_order, y_out, y_pred, y_true

        if hasattr(self, "_class_means") and self.args['model_name'] != 'coil':
            # y_order, y_out, y_true = self._eval_nme(loader, self._class_means)
            # y_pred = np.argmin(y_out, axis=1)
            # nme_accy = self._evaluate(y_order, y_pred, y_true)
            nme_accy = None
        else:
            nme_accy = None

        _save_dir = os.path.join(self.args['logfilename'], "task" + str(self._cur_task))
        os.makedirs(_save_dir, exist_ok=True)
        _pred_path = os.path.join(_save_dir, "closed_set_pred.npy")
        _target_path = os.path.join(_save_dir, "closed_set_target.npy")
        np.save(_pred_path, y_pred)
        np.save(_target_path, y_true)
        _confusion_img_path = os.path.join(_save_dir, "closed_set_conf.png")
        plot_confusion(_confusion_img_path, confusion_matrix(y_true, y_pred))
        return y_order, y_out, y_pred, y_true, cnn_accy, nme_accy

    def eval_out_task(self, type='out'):
        orders, _out_u, _pred_u, _labels_u = self.eval_task(type)
        acc_score = np.around(np.sum(_pred_u == _labels_u) * 100 / len(_labels_u), decimals=2)
        fpr, tpr, _ = roc_curve(_labels_u, _out_u[:, 1])
        auc_score = np.around(auc(fpr, tpr) * 100, decimals=2)
        _save_dir = os.path.join(self.args['logfilename'], "task" + str(self._cur_task))
        os.makedirs(_save_dir, exist_ok=True)
        _save_path = os.path.join(_save_dir, "open-set-result.csv")
        with open(_save_path, "a+") as f:
            f.write(f"{type}, {'ACC:' + str(acc_score)},{'OSCR:' + str(auc_score)}\n")
        out_cnn_accy = self._evaluate(orders, _pred_u, _labels_u)
        logging.info("open-set CNN: {}".format(out_cnn_accy["grouped"]))
        return acc_score, auc_score

    def eval_open(self, eval_type, final=False):
        self._network.eval()
        _log_dir = os.path.join("./results/", f"{self.args['prefix']}", "open-set")
        os.makedirs(_log_dir, exist_ok=True)
        out_acc, out_auc = self.eval_out_task(type='out')
        _log_path = os.path.join(_log_dir, f"{self.args['csv_name']}.csv")
        if self.args['prefix'] == 'benchmark' and final:
            with open(_log_path, "a+") as f:
                f.write(f"{self.args['time_str']}, {self.args['model_name']},")
                f.write(f"{out_acc}, {out_auc}\n")
        logging.info("open-set-out {} acc: {}, auc: {}".format(eval_type, out_acc, out_auc))
        return out_acc

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _eval(self):
        pass

    def _get_memory(self):
        if len(self._real_data_memory) == 0 and len(self._fake_data_memory):
            return None
        else:
            return (np.concatenate((self._real_data_memory, self._fake_data_memory)),
                    np.concatenate((self._real_targets_memory, self._fake_targets_memory)))

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets, order) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, type, loader):
        self._network.eval()
        y_pred, y_true, y_order = [], [], []
        print(1111)
        for i, (_, inputs, targets, orders) in enumerate(loader):
            inputs = inputs.to(self._device)
            if type == 'out' or self.args['run_type'] == 'train' or self.args['run_type'] == 'train_bak':
                with torch.no_grad():
                    outputs = self._network(inputs)["logits"]
                    # outputs, loss = self._network(inputs)
                    # outputs = self._network(inputs)
                y_order.append(orders.cpu().numpy())
                y_pred.append(outputs.cpu().numpy())
                y_true.append(targets.cpu().numpy())
            else:
                for class_id in self.args['test_class'][self._cur_task]:
                    if class_id in orders.cpu().numpy():
                        with torch.no_grad():
                            outputs = self._network(inputs)["logits"]
                            # outputs, loss = self._network(inputs)
                            # outputs = self._network(inputs)
                        y_order.append(orders.cpu().numpy())
                        y_pred.append(outputs.cpu().numpy())
                        y_true.append(targets.cpu().numpy())
        return np.concatenate(y_order), np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        y_order, real_vectors, fake_vectors, y_true = self._extract_vectors(loader)
        real_vectors = (real_vectors.T / (np.linalg.norm(real_vectors.T, axis=0) + EPSILON)).T
        fake_vectors = (fake_vectors.T / (np.linalg.norm(fake_vectors.T, axis=0) + EPSILON)).T
        real_dists = cdist(class_means[0], real_vectors, "sqeuclidean")
        fake_dists = cdist(class_means[1], fake_vectors, "sqeuclidean")
        scores = np.vstack((real_dists.T, fake_dists.T))

        return y_order, scores, y_true  # [N, topk]

    def get_curve_online(self, novel, stypes=['Bas']):
        tp, fp = dict(), dict()
        tnr_at_tpr95 = dict()
        for stype in stypes:
            if self.args['model_name'] == "icarl":
                novel = abs(np.sort(-novel))
                logging.info("model {} calculates auc using descending order!".format(self.args["model_name"]))
            else:
                novel.sort()
                logging.info("model {} calculates auc using ascending order!".format(self.args["model_name"]))
            num_n = novel.shape[0]
            print("num_n", num_n)
            tp[stype] = -np.ones([num_n + 1], dtype=int)
            fp[stype] = -np.ones([num_n + 1], dtype=int)
            tp[stype][0], fp[stype][0] = num_n, num_n
            print("tp[stype]:", tp[stype])
            print("fp[stype]:", fp[stype])
            for l in range(num_n):
                tp[stype][l + 1] = tp[stype][l]
                fp[stype][l + 1] = fp[stype][l] - 1
            tpr95_pos = np.abs(tp[stype] / num_n - .95).argmin()
            tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
        return tp, fp, tnr_at_tpr95

    def metric_ood(self, _out_u, stypes=['Bas'], verbose=True):
        if self.args['model_name'] == "icarl":
            x2 = np.min(_out_u, axis=1)
            logging.info("model {} get the minimum of the output in auc".format(self.args["model_name"]))
        else:
            x2 = np.max(_out_u, axis=1)
            logging.info("model {} get the maximum of the output in auc".format(self.args["model_name"]))
        tp, fp, tnr_at_tpr95 = self.get_curve_online(x2, stypes)
        for stype in stypes:
            # AUROC
            mtype = 'AUROC'
            tpr = np.concatenate([[1.], tp[stype] / tp[stype][0], [0.]])
            fpr = np.concatenate([[1.], fp[stype] / fp[stype][0], [0.]])
            # -np.traz(tpr, fpr)
            roc_auc = 100. * (-np.trapz(1. - fpr, tpr))
        return roc_auc

    # return order real_vec fake_vec targets
    def _extract_vectors(self, loader):
        real_vectors, fake_vectors, targets, orders = [], [], [], []
        for _, _inputs, _targets, _orders in loader:
            _targets = _targets.numpy()
            _orders = _orders.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )
            if _targets.all() == 0:
                real_vectors.append(_vectors)
            else:
                fake_vectors.append(_vectors)

            orders.append(_orders)
            targets.append(_targets)

        if len(real_vectors) == 0:
            return np.concatenate(orders), 0, np.concatenate(fake_vectors), np.concatenate(
                targets)
        elif len(fake_vectors) == 0:
            return np.concatenate(orders), np.concatenate(real_vectors), 0, np.concatenate(targets)
        else:
            return np.concatenate(orders), np.concatenate(real_vectors), np.concatenate(fake_vectors), np.concatenate(
                targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        real_dummy_data, real_dummy_targets = copy.deepcopy(self._real_data_memory), copy.deepcopy(
            self._real_targets_memory
        )
        fake_dummy_data, fake_dummy_targets = copy.deepcopy(self._fake_data_memory), copy.deepcopy(
            self._fake_targets_memory
        )
        self._class_means = np.zeros((2, self._total_classes, self.feature_dim))
        self._real_data_memory, self._real_targets_memory = self._reduce_process(real_dummy_data, real_dummy_targets,
                                                                                 np.array([]), np.array([]),
                                                                                 data_manager, m // 2, 0)
        self._fake_data_memory, self._fake_targets_memory = self._reduce_process(fake_dummy_data, fake_dummy_targets,
                                                                                 np.array([]), np.array([]),
                                                                                 data_manager, m // 2, 1)

    def _reduce_process(self, dummy_data, dummy_targets, data_memory, targets_memory, data_manager, m, label):
        for class_idx in range(self._known_classes):
            set_random(self.args['seed'])
            logging.info("Reducing exemplars for label {} data class {}".format(label, class_idx))
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            data_memory = (
                np.concatenate((data_memory, dd))
                if len(data_memory) != 0
                else dd
            )
            targets_memory = (
                np.concatenate((targets_memory, dt))
                if len(targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt), resize_size=self.args["resize_size"]
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
            )
            _, real_vectors, fake_vectors, _ = self._extract_vectors(idx_loader, oneclass=True, inference=True)
            # _, real_vectors, fake_vectors, _ = self._extract_vectors(idx_loader)
            vectors = real_vectors if not label else fake_vectors
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[label, class_idx, :] = mean
        return data_memory, targets_memory

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            set_random(self.args['seed'])
            logging.info("Constructing exemplars for class {}".format(class_idx))
            real_data, fake_data = [], []
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                resize_size=self.args["resize_size"],
                ret_data=True
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=data_manager.collate_fn
            )
            _, real_vectors, fake_vectors, _ = self._extract_vectors(idx_loader)
            for i in range(len(data)):
                if 'real' in data[i]:
                    real_data.append(data[i])
                else:
                    fake_data.append(data[i])
            real_data = np.array(real_data)
            fake_data = np.array(fake_data)

            self._construct_process(class_idx, m // 2, real_vectors, data_manager, real_data, 0)
            self._construct_process(class_idx, m // 2, fake_vectors, data_manager, fake_data, 1)

    def _construct_process(self, class_idx, per_class, vectors, data_manager, data, label):
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        class_mean = np.mean(vectors, axis=0)

        # Select
        selected_exemplars = []
        exemplar_vectors = []  # [n, feature_dim]
        for k in range(1, per_class + 1):
            S = np.sum(
                exemplar_vectors, axis=0
            )  # [feature_dim] sum of selected exemplars vectors
            mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
            i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
            selected_exemplars.append(
                np.array(data[i])
            )  # New object to avoid passing by inference
            exemplar_vectors.append(
                np.array(vectors[i])
            )  # New object to avoid passing by inference

            vectors = np.delete(
                vectors, i, axis=0
            )  # Remove it to avoid duplicative selection
            data = np.delete(
                data, i, axis=0
            )  # Remove it to avoid duplicative selection

            if len(vectors) == 0:
                break
        # uniques = np.unique(selected_exemplars, axis=0)
        # print('Unique elements: {}'.format(len(uniques)))
        selected_exemplars = np.array(selected_exemplars)
        # exemplar_targets = np.full(m, class_idx)
        exemplar_targets = np.full(selected_exemplars.shape[0], class_idx)
        set_random(self.args['seed'])
        if not label:
            self._real_data_memory = (
                np.concatenate((self._real_data_memory, selected_exemplars))
                if len(self._real_data_memory) != 0
                else selected_exemplars
            )
            self._real_targets_memory = (
                np.concatenate((self._real_targets_memory, exemplar_targets))
                if len(self._real_targets_memory) != 0
                else exemplar_targets
            )
        else:
            self._fake_data_memory = (
                np.concatenate((self._fake_data_memory, selected_exemplars))
                if len(self._fake_data_memory) != 0
                else selected_exemplars
            )
            self._fake_targets_memory = (
                np.concatenate((self._fake_targets_memory, exemplar_targets))
                if len(self._fake_targets_memory) != 0
                else exemplar_targets
            )

        # Exemplar mean
        idx_dataset = data_manager.get_dataset(
            [],
            source="train",
            mode="test",
            appendent=(selected_exemplars, exemplar_targets),
            resize_size=self.args["resize_size"]
        )
        idx_loader = DataLoader(
            idx_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        _, real_vectors, fake_vectors, _ = self._extract_vectors(idx_loader, oneclass=True, inference=True)
        vectors = real_vectors if not label else fake_vectors
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        mean = np.mean(vectors, axis=0)
        mean = mean / np.linalg.norm(mean)

        self._class_means[label, class_idx, :] = mean
