import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from test import evaluate
from utils import save_checkpoint, resume_from_checkpoint, \
    load_data, build_model_optm, get_train_args, fix_base_para
from loss import CrossMatchingTripletLoss, WeightedCrossMatchingTripletLoss
from logger import Logger
from trainer import train, validate


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(passed_args=None):
    global args
    # args = parser.parse_args()
    args = get_train_args(passed_args)

    # generate exp name
    args.savedir = '{}/{}/{}/exp-{}-{}-m{}-TW{}-it{}-b{}-f{}/'.format(
        args.savedir, args.arch, args.dataset, args.remarks, args.loss, args.margin, args.tri_lambda,
        args.epoch_lenth, args.batch_size, args.num_hashing)
    args.resume_file = 'model_best.pth.tar'
    args.precision = True
    args.recompute = False
    args.pretrained = False
    if args.dataset == 'sketchy2':
        args.dataset = 'sketchy'
        args.zero_version = 'zeroshot2'

    sys.stdout = Logger(os.path.join(args.savedir, 'log.txt'))

    print(time.strftime('train-%Y-%m-%d-%H-%M-%S'))
    print(args)

    criterion_train = nn.CrossEntropyLoss()
    criterion_train_t = CrossMatchingTripletLoss(margin=args.margin,
                                                 normalize_feature=True,
                                                 mode='basic')

    if args.loss == 'ce':
        args.tri_lambda = 0
    elif args.loss == 'cross':
        criterion_train_t = CrossMatchingTripletLoss(margin=args.margin,
                                                     normalize_feature=True,
                                                     mode='basic')
    elif args.loss == 'within':
        criterion_train_t = CrossMatchingTripletLoss(margin=args.margin,
                                                     normalize_feature=True,
                                                     mode='within')
    elif args.loss == 'hybrid':
        criterion_train_t = CrossMatchingTripletLoss(margin=args.margin,
                                                     normalize_feature=True,
                                                     mode='partial')
    elif args.loss == 'all':
        criterion_train_t = CrossMatchingTripletLoss(margin=args.margin,
                                                     normalize_feature=True,
                                                     mode='all')
    elif args.loss == 'mathm':
        criterion_train_t = WeightedCrossMatchingTripletLoss(margin=args.margin,
                                                             normalize_feature=True,
                                                             mode='all')

    # elif args.loss == 'ctri':
    #     args.ds_tri = False
    #     criterion_train_t = CrossMatchingTripletLoss(margin=args.margin,
    #                                                  normalize_feature=True,
    #                                                  mode=args.cross_mode)
    # elif args.loss == 'wctri':
    #     criterion_train_t = WeightedCrossMatchingTripletLoss(margin=args.margin,
    #                                                          normalize_feature=True,
    #                                                          mode=args.cross_mode)

    model, optimizer, scheduler = build_model_optm(args)

    optimizer.add_param_group({'params': list(criterion_train.parameters())})
    # scheduler.optimizer = optimizer

    resume_from_checkpoint(model, os.path.join(args.resume_dir, 'checkpoint.pth.tar'))

    cudnn.benchmark = True

    mix_loader, val_loader = load_data(args)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    best_acc1 = 0
    print('start training')
    for epoch in range(args.epochs):

        torch.cuda.empty_cache()
        if epoch < args.fixbase_epochs:    # fix pretrained para in first few epochs
            fix_base_para(model)
        else:
            model.requires_grad_(True)

        print(epoch, *[param_group['lr'] for param_group in optimizer.param_groups])
        try:
            # model_t = None
            train(mix_loader, model, criterion_train, criterion_train_t,
                  optimizer, epoch, args)
        except RuntimeError as e:
            raise e
            # print(e)
        acc1 = validate(val_loader, model, args)

        scheduler.step()
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=os.path.join(args.savedir, 'checkpoint.pth.tar'))
        if (epoch + 1) % args.eval_period == 0:
            torch.cuda.empty_cache()
            evaluate(args, args.savedir, get_precision=True, model=model, recompute=True)

    args.itq = True
    # args.num_q_hashing = 64
    print('\n\n ----------------------  eval with itq -------------------------\n\n')
    evaluate(args, args.savedir, get_precision=True, model=model, recompute=False)


if __name__ == '__main__':

    main()
