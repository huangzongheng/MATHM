import torch
from apex import amp
import os
import argparse
import os.path as osp
import errno
import math
from Datasets import SketchyDataset, TUBerlinDataset
from torch.utils.data import DataLoader
from torch import nn
import functools
import torchvision.transforms as transforms
from mix_dataset import MixDatasets, MutiSourceRandomIdentitySampler, IterLoader
import datetime
from ResnetModel import HashingModel
import pretrainedmodels

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if name.islower() and not name.startswith("__"))


def get_train_args(passed_args=None):
    parser = argparse.ArgumentParser(description='PyTorch CSE_ResNet Model for TUBerlin Training')

    parser.add_argument('--savedir', '-s',  metavar='DIR',
                        default='../logs',
                        help='path to save dir')
    parser.add_argument('--resume-dir',
                        default=' ',
                        type=str, metavar='PATH',
                        help='dir of model checkpoint (default: none)')
    parser.add_argument('--resume-file',
                        default='checkpoint.pth.tar',   # model_best.pth.tar
                        type=str, metavar='PATH',
                        help='file name of model checkpoint (default: none)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='cse_resnet50',
                        # choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: cse_resnet50)')
    parser.add_argument('--dataset', '-d',
                        default='tuberlin',
                        type=str, choices=['sketchy', 'sketchy2', 'tuberlin'],
                        help='dir of model checkpoint (default: none)')
    parser.add_argument('--remarks',  metavar='str',
                        default='',
                        help='tag for this experiment')
    parser.add_argument('--num-classes', metavar='N', type=int, default=220,
                        help='number of classes (default: 220)')
    parser.add_argument('--num-hashing', metavar='N', type=int, default=512,
                        help='number of hashing dimension (default: 512)')
    parser.add_argument('--num-q-hashing', metavar='N', type=int, default=64,
                        help='number of ITQ hashing dimension (default: 64)')
    parser.add_argument('--num-workers', metavar='N', type=int, default=4,
                        help='number threads to load data')

    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--epoch-lenth', default=200, type=int, metavar='N',
                        help='iterations per epoch')
    parser.add_argument('--fixbase-epochs', default=2, type=int, metavar='N',
                        help='epochs for fixbase training')
    parser.add_argument('--batch-size', '-b', default=128, type=int, metavar='N',
                        help='number of samples per batch')
    parser.add_argument('--num-instance', '-n', default=8, type=int, metavar='N',
                        help='number of img per class')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate (default: 0.01)')
    parser.add_argument('--margin',default=0.2, type=float,
                        help='margin for triplet loss')
    parser.add_argument('--base-lr-mult', default=0.1, type=float,
                        metavar='LR', help='decay factor for backbone params')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--eval-period', default=10, type=int,
                        metavar='N', help='evaluate frequency (default: 10)')
    # parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
    #                     help='evaluate model on validation set')
    # parser.add_argument('-f', '--freeze_features', dest='freeze_features', action='store_true',
    #                     help='freeze features of the base network')
    parser.add_argument('--tri-lambda', metavar='LAMBDA', default='0.0', type=float,
                        help='lambda for triplet loss (default: 1)')
    parser.add_argument('--zero-version', metavar='VERSION', default='zeroshot', type=str,
                        help='zeroshot version for training and testing (default: zeroshot)')

    parser.add_argument('--loss', metavar='LAMBDA', default='ce', type=str,
                        help='loss type')
    parser.add_argument('--cross-mode', metavar='LAMBDA', default='all', type=str,
                        help='cross tri type')
    parser.add_argument('--cse-end', default=4, type=int,
                        metavar='N', help='layers that use cse block (default: 4)')
    parser.add_argument('--lr-sch', metavar='LAMBDA', default='cos', type=str,
                        help='zeroshot version for training and testing (default: zeroshot)')
    parser.add_argument('--warmup', metavar='LAMBDA', default=5, type=int,
                        help='warmup epochs')
    # testing args
    parser.add_argument('--pretrained', action='store_true', help='evaluate the pretrained model')
    parser.add_argument('--visualize', action='store_true', help='visualize rank result')
    parser.add_argument('--itq', action='store_true', help='use itq to binary feature')
    parser.add_argument('--precision', action='store_true', help='report precision@100')
    parser.add_argument('--recompute', action='store_true', help='re-extract feature for testing')

    if passed_args is None:
        return parser.parse_args()
    else:
        return parser.parse_args(passed_args)


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    # if is_best:
    #     filepath = '/'.join(filename.split('/')[0:-1])
    #     shutil.copyfile(filename, os.path.join(filepath,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def resume_from_checkpoint(model, path):

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location='cpu')
        # args.start_epoch = checkpoint['epoch']

        save_dict = checkpoint['state_dict']
        model_dict = model.state_dict()

        trash_vars = [k for k in save_dict.keys() if k not in model_dict.keys()]
        print('trashed vars from resume dict:')
        print(trash_vars)

        resume_dict = {k: v for k, v in save_dict.items() if k in model_dict}

        model_dict.update(resume_dict)
        print(model.load_state_dict(model_dict, strict=False))

        # model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(path))
        # return


def adjust_learning_rate(epoch, warmup_epochs=3, mode='exp', max_epoch=20, fix_epochs=0):
    lr = 1
    assert fix_epochs < max_epoch - warmup_epochs
    epoch -= fix_epochs
    max_epoch -= fix_epochs
    if epoch < 0:
        lr = lr                 # fix base
    elif epoch < warmup_epochs:
        lr = lr * (0.01 + epoch / warmup_epochs)   # warmup
    elif epoch >= warmup_epochs * 2:
        if mode == 'cos':
            lr = lr * (1 + math.cos(math.pi*(epoch - 1*warmup_epochs) / (max_epoch - 1*warmup_epochs)))/2
        elif mode == 'exp':
            lr = lr * math.pow(0.001, float(epoch - warmup_epochs) / (max_epoch - warmup_epochs))     # exp decay
        else:  # if mode == 'const':
            pass
    else:
        pass

    return lr


def load_data(args):
    transformations = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
    ])
    sample_rate = (0.5, 0.5)

    if args.dataset == 'tuberlin':
        skt_train = TUBerlinDataset(split='train', zero_version = args.zero_version,
                                         transform=transformations, aug='sketch', cid_mask = True)
        pht_train = TUBerlinDataset(split='train', zero_version = args.zero_version,
                                             version='ImageResized_ready',
                                             transform=transformations, aug='img', cid_mask = True)
        skt_val = TUBerlinDataset(split='val', zero_version = args.zero_version,
                                       transform=transformations, aug=False)

    elif args.dataset == 'sketchy':
        if args.zero_version == 'zeroshot2':
            pass
        else:
            args.zero_version = 'zeroshot1'
        skt_train = SketchyDataset(split='train', zero_version=args.zero_version,
                                        transform=transformations, aug='sketch', cid_mask = True)
        pht_train = SketchyDataset(split='train', version='all_photo', zero_version=args.zero_version,
                                            transform=transformations, aug='img', cid_mask = True)
        skt_val = SketchyDataset(split='val', zero_version=args.zero_version,
                                      transform=transformations, aug=False)

    else:
        print('not support dataset', args.dataset)

    mix_train = MixDatasets(skt_train, pht_train)
    mix_loader = DataLoader(dataset=mix_train,
                            batch_size=args.batch_size,
                            sampler=MutiSourceRandomIdentitySampler(mix_train, args.batch_size,
                                                                    args.num_instance, p=sample_rate,
                                                                    epoch_lenth=args.epoch_lenth),
                            num_workers=args.num_workers)
    mix_loader = IterLoader(mix_loader, args.epoch_lenth)

    val_loader = DataLoader(dataset=skt_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    print(str(datetime.datetime.now()) + ' data loaded.')
    return mix_loader, val_loader


def build_model_optm(args):
    if args.dataset == 'tuberlin':
        args.num_classes = 220

    elif args.dataset == 'sketchy':
        if args.zero_version == 'zeroshot2':
            args.num_classes = 104
        else:
            args.num_classes = 100

    model = HashingModel(args.arch, args.num_hashing, args.num_classes)

    model = model.to(device)
    print(str(datetime.datetime.now()) + 'model inited.')

    base_params, new_params = get_new_params(model, True)

    param_groups = [
        {'params': base_params, 'lr': args.lr * args.base_lr_mult},
        {'params': new_params},
    ]

    optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    lr_generator = functools.partial(
        adjust_learning_rate, mode=args.lr_sch, warmup_epochs=args.warmup, max_epoch=args.epochs,
        fix_epochs=args.fixbase_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_generator)

    if 'cuda' in device:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', enabled=True)
    model = nn.DataParallel(model)

    return model, optimizer, scheduler


def fix_base_para(model):
    model.requires_grad_(True)
    print('training with base paramaters fixed')
    base_params, new_params = get_new_params(model)
    for para in base_params:
        para.requires_grad_(False)


def get_new_params(model, verbose=False, ignored_para=('second',)):
    base_params = []
    new_params = []
    for name, para in model.named_parameters():
        # if name contains ignored words, ignore this parameter.
        # if len(ignored_para) > 0 and sum([ig in name for ig in ignored_para]) > 0:
        #     continue

        if ('fc_tag' in name or 'linear' in name) and 'last' not in name:    # new parameters: CSE_fc, classifier
            if verbose:
                print(name)
            new_params.append(para)
        else:
            base_params.append(para)

    return base_params, new_params
