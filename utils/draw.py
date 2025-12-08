import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix


def evaluate_multiclass(gt_labels, pred_labels):
    acc = accuracy_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels, average='macro')
    recall = recall_score(gt_labels, pred_labels, average='macro')
    recalls = recall_score(gt_labels, pred_labels, average=None)  # 每一类recall返回
    return {'recalls': recalls, 'recall': recall, 'f1': f1, 'acc': acc}


def plot_confusion(filename, CM):
    plt.matshow(CM, cmap=plt.cm.Reds)
    plt.ylabel('True label', fontdict={'size': 20})
    plt.xlabel('Predicted label', fontdict={'size': 20})
    plt.colorbar()
    plt.savefig(filename, dpi=300)
    plt.close()


if __name__ == "__main__":
    pred = np.load("../pred.npy")
    target = np.load("../target.npy")
    results = evaluate_multiclass(target, pred)
    CM = confusion_matrix(target, pred)
    plot_confusion('', CM)
