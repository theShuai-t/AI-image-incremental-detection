import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import defaultdict

from utils.toolkit import set_random


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def swap_spe_features(type_list, value_list):
    type_list = type_list.cpu().numpy().tolist()
    # get index
    index_list = list(range(len(type_list)))

    # init a dict, where its key is the type and value is the index
    spe_dict = defaultdict(list)

    # do for-loop to get spe dict
    for i, one_type in enumerate(type_list):
        spe_dict[one_type].append(index_list[i])

    # shuffle the value list of each key
    for keys in spe_dict.keys():
        random.shuffle(spe_dict[keys])

    # generate a new index list for the value list
    new_index_list = []
    for one_type in type_list:
        value = spe_dict[one_type].pop()
        new_index_list.append(value)

    # swap the value_list by new_index_list
    value_list_new = value_list[new_index_list]

    return value_list_new


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def contrastive_loss(self, anchor, positive, negative):
        dist_pos = F.pairwise_distance(anchor, positive)
        dist_neg = F.pairwise_distance(anchor, negative)
        # Compute loss as the distance between anchor and negative minus the distance between anchor and positive
        loss = torch.mean(torch.clamp(dist_pos - dist_neg + self.margin, min=0.0))
        return loss

    def forward(self, common, specific, spe_label):
        # prepare
        bs = common.shape[0]
        real_common, fake_common = common.chunk(2)
        ### common real
        idx_list = list(range(0, bs // 2))
        random.shuffle(idx_list)
        real_common_anchor = common[idx_list]
        ### common fake
        idx_list = list(range(bs // 2, bs))
        random.shuffle(idx_list)
        fake_common_anchor = common[idx_list]
        ### specific
        specific_anchor = swap_spe_features(spe_label, specific)
        real_specific_anchor, fake_specific_anchor = specific_anchor.chunk(2)
        real_specific, fake_specific = specific.chunk(2)

        # Compute the contrastive loss of common between real and fake
        loss_realcommon = self.contrastive_loss(real_common, real_common_anchor, fake_common_anchor)
        loss_fakecommon = self.contrastive_loss(fake_common, fake_common_anchor, real_common_anchor)

        # Comupte the constrastive loss of specific between real and fake
        loss_realspecific = self.contrastive_loss(real_specific, real_specific_anchor, fake_specific_anchor)
        loss_fakespecific = self.contrastive_loss(fake_specific, fake_specific_anchor, real_specific_anchor)

        # Compute the final loss as the sum of all contrastive losses
        loss = loss_realcommon + loss_fakecommon + loss_fakespecific + loss_realspecific
        return loss


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, feat_dim=768, device=None):
        super(CenterLoss, self).__init__()
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(1, self.feat_dim).to(device))
        # self.radius = torch.tensor(0.0).to(device)
        nn.init.kaiming_uniform_(self.centers)

    def forward(self, x):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, 1) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(1, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        loss = distmat.clamp(min=1e-12, max=1e+12)
        # dist = torch.sum((x - self.centers) ** 2, dim=1)
        # scores = dist - self.radius ** 2
        # loss = self.radius ** 2 + 10 * torch.mean(torch.max(torch.zeros_like(scores), scores))

        return loss

class MultiCenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, feat_dim=768, num_classes=1, device=None):
        super(MultiCenterLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.device = device
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        # self.radius = nn.Parameter(torch.zeros(1).to(device))
        nn.init.kaiming_uniform_(self.centers)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss




class SphereLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, feat_dim=768, device=None):
        super(SphereLoss, self).__init__()
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(1, self.feat_dim).to(device))
        self.radius = nn.Parameter(torch.randn(1).to(device))
        nn.init.kaiming_uniform_(self.centers)
        nn.init.kaiming_uniform_(self.radius)

    def forward(self, x):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, 1) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(1, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        dist = distmat.clamp(min=1e-12, max=1e+12)
        loss = self.radius ** 2 + torch.mean(torch.max(torch.zeros_like(dist), dist-self.radius ** 2))
        return loss