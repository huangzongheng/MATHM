"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils import model_zoo
import numpy as np
from global_tag import DTag

    
pretrained_settings = {
    'senet154': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet101': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet152': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext50_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
        #                      padding=0)
        # self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
        #                      padding=0)


        self.fc1 = nn.Linear(in_features=channels, out_features=channels // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=channels // reduction, out_features=channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y=None):
        module_input = x
        x = self.avg_pool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(x.size()[0], x.shape[1], 1, 1)
        return module_input * x


class CSEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(CSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(in_features=channels, out_features=channels // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(in_features=channels // reduction+1, out_features=channels, bias=True)
        self.fc2 = nn.Linear(in_features=channels // reduction, out_features=channels, bias=True)
        self.fc_tag = nn.Linear(in_features=1, out_features=channels, bias=False)       #分离tag weight
        nn.init.constant_(self.fc_tag.weight, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        module_input = x
        x = self.avg_pool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc1(x)
        x = self.relu(x)
        
        # x = torch.cat((x, y), 1)
        # x = self.fc2(x)
        x = self.fc2(x) + self.fc_tag(y)       # * x.shape[1] 增强CSE模块的影响力
        x = self.sigmoid(x)
        
        x = x.view(x.size()[0], x.shape[1], 1, 1)
        
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out

        
class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x, y=None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if y is None:
            y = DTag.domain_tag[:, None].to(out.dtype)

        if self.downsample is not None:
            residual = self.downsample(x)

        if x.shape[0] != y.shape[0]:        # 对不共享的部分模型，channel数会不一致，这里直接忽略domain tag输入
            y = torch.zeros(out.shape[0], 1, device=out.device, dtype=out.dtype)

        out = self.se_module(out, y) + residual
        out = self.relu(out)

        return out


class CSEResNetBottleneck(SEResNetBottleneck):
    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super().__init__(inplanes, planes, groups, reduction, stride, downsample)
        self.se_module = CSEModule(planes * 4, reduction=reduction)


class CSENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000, ems=False):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For CSE-ResNet models: CSEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(CSENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.ems = ems
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x, y):
        x = self.layer0(x)
        for m in self.layer1._modules.values():
            x = m(x, y)
            
        for m in self.layer2._modules.values():
            x = m(x, y)
            
        for m in self.layer3._modules.values():
            x = m(x, y)
            
        for m in self.layer4._modules.values():
            x = m(x, y)
            
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x, y):
        x = self.features(x, y)
        x = self.logits(x)
        return x


class CSENet_hashing(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000, ems=False, hashing_dim=64,):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For CSE-ResNet models: CSEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(CSENet_hashing, self).__init__()
        if isinstance(block, nn.Module):
            block = [block]*4
        assert len(block) == 4
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        # layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
        #                                             ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(
            block[0],
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block[1],
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block[2],
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block[3],
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        block = block[-1]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.ems = ems
        self.second_last_linear = nn.Linear(512 * block.expansion, hashing_dim)
        self.last_linear = nn.Linear(hashing_dim, num_classes, bias=False)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x, y, return_pyramid=False):
        pyramid = []
        DTag.set_domain_tag(y.int().squeeze())
        x = self.layer0(x)
        x = self.maxpool(x)
        # for m in self.layer1._modules.values():
        #     x = m(x, y)
        #
        # for m in self.layer2._modules.values():
        #     x = m(x, y)
        #
        # for m in self.layer3._modules.values():
        #     x = m(x, y)
        #
        # for m in self.layer4._modules.values():
        #     x = m(x, y)
        x = self.layer1(x)
        pyramid.append(x)
        x = self.layer2(x)
        pyramid.append(x)
        x = self.layer3(x)
        pyramid.append(x)
        x = self.layer4(x)
        pyramid.append(x)

        if return_pyramid:
            x = x, pyramid
            
        return x

    def hashing(self, x):
        x = self.avg_pool(x)
        # x = (self.avg_pool(x) + self.max_pool(x)) / 2 # add max pool to concentrate on small objects
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.second_last_linear(x)
        return x
    
    def logits(self, x):
        x = self.last_linear(x)
        return x

    def forward(self, x, y):
        x = self.features(x, y)
        x = self.hashing(x)
        x = self.logits(x)
        return x
    
    

def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    

def initialize_pretrained_model_ext(model, num_classes, settings):
    pretrained_state_dict = model_zoo.load_url(settings['url'])
    for kk,vv in pretrained_state_dict.items():
        if 'se_module.fc1.weight' in kk:
            pretrained_state_dict[kk] = torch.squeeze(vv)
        if 'se_module.fc2.weight' in kk:
            pretrained_state_dict[kk] = torch.squeeze(vv)       # 分离tag weight
            # pretrained_state_dict[kk] = torch.cat([torch.squeeze(vv), torch.zeros(vv.size()[0], 1)], dim=1)
    
    if num_classes != settings['num_classes'] or model.ems:
        del(pretrained_state_dict['last_linear.weight'])
        del(pretrained_state_dict['last_linear.bias'])
        
    model_dict = model.state_dict()
    trash_vars = [k for k in pretrained_state_dict.keys() if k not in model_dict.keys()]
    print('trashed vars from resume dict:')
    print(trash_vars)

    resume_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}

    model_dict.update(resume_dict)
    model.load_state_dict(model_dict)
    
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    
    
def initialize_pretrained_model_hashing(model, hashing_dim, num_classes, settings):
    pretrained_state_dict = model_zoo.load_url(settings['url'])
    for kk,vv in pretrained_state_dict.items():
        if 'se_module.fc1.weight' in kk:
            pretrained_state_dict[kk] = torch.squeeze(vv)
        if 'se_module.fc2.weight' in kk:
            pretrained_state_dict[kk] = torch.squeeze(vv)       # 分离tag weight
            # pretrained_state_dict[kk] = torch.cat([torch.squeeze(vv), torch.zeros(vv.size()[0], 1)], dim=1)
    #pdb.set_trace()
    old_weight = pretrained_state_dict['last_linear.weight']
    print(old_weight.shape)
    u, s, vh = np.linalg.svd(old_weight, full_matrices=True)
#    u_new = u[:hashing_dim,:hashing_dim]
    s_new = np.diag(s)[:hashing_dim,:hashing_dim]
    s_new = s_new * (s.sum() / s_new.sum())         # 保证logits幅度不变
#    vh_new = vh[:hashing_dim,:]

    new_1 = u[:, :hashing_dim]
    new_2 = np.dot(s_new, vh[:hashing_dim, :])
#    new_weight = np.dot(u_new, np.dot(s_new, vh_new))
    print(new_1.shape, new_2.shape)
    
    del(pretrained_state_dict['last_linear.weight'])
    del(pretrained_state_dict['last_linear.bias'])
    
    model_dict = model.state_dict()
    trash_vars = [k for k in pretrained_state_dict.keys() if k not in model_dict.keys()]
    print('trashed vars from resume dict:')
    print(trash_vars)

    # 旋转SVD后的权向量，平均分布能量
    proj = torch.empty(hashing_dim,hashing_dim)
    nn.init.orthogonal_(proj)

    resume_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
    resume_dict['second_last_linear.weight'] = proj @ torch.from_numpy(new_2)
    resume_dict['last_linear.weight'] = torch.from_numpy(new_1) @ proj.t()

    model_dict.update(resume_dict)
    model.load_state_dict(model_dict)
    
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def cse_resnet50(num_classes=1000, pretrained='imagenet', ems=False):
    model = CSENet(CSEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes, ems=ems)
    
    if pretrained is not None:
        settings = pretrained_settings['se_resnet50'][pretrained]
        initialize_pretrained_model_ext(model, num_classes, settings)
    return model


def cse_resnet50_hashing(hashing_dim, num_classes=1000, pretrained='imagenet', ems=False, cse_end=4):
    assert 0 <= cse_end <= 4
    blocks = [CSEResNetBottleneck] * cse_end + [SEResNetBottleneck] * (4-cse_end)
    model = CSENet_hashing(blocks, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes, ems=ems, hashing_dim=hashing_dim)
    
    if pretrained is not None:
        # settings = pretrained_settings['se_resnet50'][pretrained]
        settings = pretrained_settings['se_resnet50']['imagenet']
        initialize_pretrained_model_hashing(model, hashing_dim, num_classes, settings)
    return model
