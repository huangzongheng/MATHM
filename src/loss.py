import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cross_modal_hard_example_mining(dist_mat, labels, tags, mode='all', return_ind=False):

    assert len(dist_mat.size()) == 2
    is_pos = labels[:, None].eq(labels)
    is_neg = labels[:, None].ne(labels)
    is_same_modal = tags[:, None].eq(tags)
    N, M = dist_mat.shape
    dist_ap, relative_p_inds = torch.max(dist_mat - (is_neg + ~is_same_modal)*1000, 1, keepdim=True)
    dist_an, relative_n_inds = torch.min(dist_mat + (is_pos + ~is_same_modal) * 1000, 1, keepdim=True)
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    dist_apc, relative_pc_inds = torch.max(dist_mat - (is_neg + is_same_modal)*1000, 1, keepdim=True)
    dist_anc, relative_nc_inds = torch.min(dist_mat + (is_pos + is_same_modal) * 1000, 1, keepdim=True)
    dist_apc = dist_apc.squeeze(1)
    dist_anc = dist_anc.squeeze(1)

    # the detached term is added to make sure the gradient of each loss won't change when combining multiple losses
    if mode == 'basic':
        dist_ap = [dist_ap, dist_ap.detach(), dist_ap.detach()]
        dist_an = [dist_an, dist_an.detach(), dist_an.detach()]
    elif mode == 'within':
        dist_ap = [dist_apc, dist_apc.detach(), dist_apc.detach()]
        dist_an = [dist_anc, dist_anc.detach(), dist_anc.detach()]
    elif mode == 'partial':
        dist_ap = [dist_apc, dist_apc.detach(), dist_apc.detach()]
        dist_an = [dist_an, dist_an.detach(), dist_an.detach()]
    elif mode == 'all':
        dist_ap = [dist_ap, dist_apc, dist_apc]
        dist_an = [dist_an, dist_an, dist_anc]

    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)

    if return_ind is False:
        return dist_ap, dist_an
    else:
        pind = torch.cat([relative_p_inds, relative_pc_inds, relative_p_inds, relative_pc_inds, ])
        nind = torch.cat([relative_n_inds, relative_n_inds, relative_nc_inds, relative_nc_inds, ])
        return dist_ap, dist_an, pind, nind


class CrossMatchingTripletLoss(nn.Module):
    def __init__(self, margin=0, normalize_feature=False, mode='all', share_neg=True):
        super().__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        # assert mode in ['all', 'same', 'cross']
        assert mode in ['basic', 'within', 'partial', 'all']
        self.mode = mode
        self.share_neg = share_neg

    @staticmethod
    def get_dist(feat1, feat2):
        return euclidean_dist(feat1, feat2)

    def forward(self, global_feat, labels, tags):

        if self.normalize_feature:
            global_feat = F.normalize(global_feat, dim=-1)
        dist_mat = self.get_dist(global_feat, global_feat)

        dist_ap, dist_an = cross_modal_hard_example_mining(
            dist_mat, labels, tags, self.mode)

        loss = torch.relu(dist_ap - dist_an + self.margin)
        return loss.mean(-1).sum(), dist_ap, dist_an


class WeightedCrossMatchingTripletLoss(CrossMatchingTripletLoss):

    beta = 20

    def forward(self, global_feat, labels, tags):
        assert self.mode == 'all'
        _, dist_ap, dist_an = super().forward(global_feat, labels, tags)

        dist = dist_ap - dist_an + self.margin

        dist = dist.reshape(-1, labels.shape[0])

        # soft relu for better gradient estimation
        loss = F.softplus(dist, beta=self.beta).mean(-1)

        # gradient for softplus
        loss_grad = torch.sigmoid(dist*self.beta).mean(-1)

        # gradient-based weighting
        weight = loss_grad.mean() / loss_grad.clamp_min(0.1)

        loss = loss * weight[:, None]
        loss = loss.sum()

        return loss, dist_ap, dist_an


