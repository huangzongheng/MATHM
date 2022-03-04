import torch
import torch.nn as nn


# global domain tag
class DTag(nn.Module):

    domain_tag = None

    @staticmethod
    def set_domain_tag(tag):
        DTag.domain_tag = tag
