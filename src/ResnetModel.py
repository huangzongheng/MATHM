import torch
import torch.nn as nn
from senet import cse_resnet50_hashing
from resnet import resnet50_hashing
from global_tag import DTag


class HashingModel(nn.Module):
    def __init__(self, arch, hashing_dim, num_classes, pretrained=True, cse_end=4):
        super(HashingModel, self).__init__()

        self.hashing_dim = hashing_dim
        self.num_classes = num_classes
        self.modelName = arch

        if self.modelName == 'resnet50':
            self.original_model = resnet50_hashing(self.hashing_dim, pretrained=pretrained)
        elif self.modelName == 'se_resnet50':
            self.original_model = cse_resnet50_hashing(self.hashing_dim, cse_end=0, pretrained=pretrained)
        elif self.modelName == 'cse_resnet50':
            self.original_model = cse_resnet50_hashing(self.hashing_dim, cse_end=cse_end, pretrained=pretrained)

        self.original_model.last_linear = nn.Sequential()
        self.linear = nn.Linear(in_features=hashing_dim, out_features=num_classes, bias=False)

    def forward(self, x, y, return_feat=False):
        DTag.set_domain_tag(y.int().squeeze())
        feats = self.original_model.features(x, y)
        feats = self.original_model.hashing(feats)

        out = self.linear(feats)
        if return_feat:
            return out, feats
        else:
            return out



