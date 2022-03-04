import torch
import torch.nn as nn
import time
import numpy as np
from apex import amp
from utils import accuracy, save_checkpoint, AverageMeter

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(train_loader, model, criterion, criterion_train_t, \
               optimizer, epoch, args):

    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_t = AverageMeter()
    avg_s_ap = AverageMeter()
    avg_s_an = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    s_ap = torch.tensor(0., device=device)
    s_an = torch.tensor(0., device=device)


    # switch to train mode
    model.train()
    end = time.time()

    for i, (input_all, target_all, tag_all) in enumerate(train_loader):
        input_all = input_all.to(device)
        tag_all = tag_all.to(device)
        target_all = target_all.type(torch.LongTensor).view(-1,).to(device)

        optimizer.zero_grad()

        output, feat = model(input_all, tag_all, return_feat=True)

        loss = criterion(output, target_all)

        # local metric loss
        if args.tri_lambda > 0:
            loss_t, s_ap, s_an = criterion_train_t(feat, target_all, tag_all)
        else:
            loss_t = 0 * loss


        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target_all, topk=(1, 5))
        losses.update(loss.item(), input_all.size(0))
        losses_t.update(loss_t.item(), input_all.size(0))
        avg_s_ap.update(s_ap.mean().item(), input_all.size(0))
        avg_s_an.update(s_an.mean().item(), input_all.size(0))
        top1.update(acc1[0], input_all.size(0))
        top5.update(acc5[0], input_all.size(0))

        # compute gradient and do SGD step
        loss_total = loss + args.tri_lambda*loss_t

        if 'cuda' in device:
            with amp.scale_loss(loss_total, optimizer) as loss_total:
                loss_total.backward()
        else:
            loss_total.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % args.print_freq == 0 or i == len(train_loader)-1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Loss {loss.val:.3f} {loss_t.val:.3f} '
                  '({loss.avg:.3f} {loss_t.avg:.3f} sp:{s_ap.avg:.3f} sn:{s_an.avg:.3f}) '
                  'Acc@1 {top1.val:.2f} ({top1.avg:.2f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, loss_t=losses_t,
                s_ap=avg_s_ap, s_an=avg_s_an,
                top1=top1))


def validate(val_loader, model, args):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # model_t.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.type(torch.LongTensor).view(-1,)
        target = target.to(device)

        # compute output
        with torch.no_grad():
            output = model(input, torch.zeros(input.size()[0],1).to(device))

        # measure accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(val_loader)-1:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Acc@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                i, len(val_loader), batch_time=batch_time,
                top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

