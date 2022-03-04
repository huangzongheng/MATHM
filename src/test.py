import sys
sys.path.append('.')
import argparse
import os
import pickle
from Datasets import SketchyDataset, TUBerlinDataset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import datetime
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.distance import cdist
import pretrainedmodels
import torch.nn.functional as F
from ResnetModel import HashingModel
from logger import Logger
from utils import resume_from_checkpoint
from visualize import visualize_ranked_results
from itq import compressITQ
from utils import load_data, get_train_args

# warnings.filterwarnings("error")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate(args, resume_dir, get_precision=False, model=None, recompute=False, visualize=False):
    args.resume_dir = resume_dir

    for phase in ['zero', 'photo', 'sketch']:
        print('evaluating {} based image retrival result'.format(phase))
        feature_file = os.path.join(resume_dir, 'features_{}.pickle'.format(phase))
        if os.path.isfile(feature_file) and not recompute:
            print('load saved SBIR features')
            with open(feature_file, 'rb') as fh:
                predicted_features_gallery, gt_labels_gallery, \
                predicted_features_query, gt_labels_query, \
                scores = pickle.load(fh)
            if args.itq:
                predicted_features_gallery, predicted_features_query = \
                    compressITQ(predicted_features_gallery, predicted_features_query, q_dim=args.num_q_hashing)
                scores = - cdist(predicted_features_query, predicted_features_gallery, metric='hamming')

            if scores is None:
                scores = - cdist(predicted_features_query, predicted_features_gallery)
        elif phase is 'zero':
            print('prepare SBIR features using saved model')
            predicted_features_gallery, gt_labels_gallery, \
            predicted_features_query, gt_labels_query, \
            scores, datasets = prepare_features(args, model, args.itq)

            # visualize
            if visualize:
                visualize_ranked_results(scores, datasets, save_dir=os.path.join(args.resume_dir, 'visualize'), topk=10)
        elif phase is 'photo':
            print('prepare PBIR features')
            predicted_features_gallery, gt_labels_gallery, \
            predicted_features_query, gt_labels_query, \
            scores = prepare_pbir_features(predicted_features_gallery, gt_labels_gallery, resume_dir, args.itq)
        else:
            print('prepare SBSR features')
            with open(os.path.join(resume_dir, 'features_zero.pickle'), 'rb') as fh:
                _,_,sfeat,slabel,_ = pickle.load(fh)
            predicted_features_gallery, gt_labels_gallery, \
            predicted_features_query, gt_labels_query, \
            scores = prepare_sbsr_features(sfeat, slabel, resume_dir, args.itq)

        mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
        for fi in range(predicted_features_query.shape[0]):
            mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery)
            mAP_ls[gt_labels_query[fi]].append(mapi)

        print('calculating average mAP')
        for mAPi,mAPs in enumerate(mAP_ls):
            print(str(mAPi)+' '+str(np.nanmean(mAPs))[:5]+' '+str(np.nanstd(mAPs))[:5])
        all_AP = sum(mAP_ls, [])
        print('Average mAP: {} {}'.format(str(np.nanmean(all_AP))[:5], str(np.nanstd(all_AP))[:5]))

        if get_precision:

            prec_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
            for fi in range(predicted_features_query.shape[0]):
                prec = eval_precision(gt_labels_query[fi], scores[fi], gt_labels_gallery)
                prec_ls[gt_labels_query[fi]].append(prec)
            print('calculating average precision')

            for preci,precs in enumerate(prec_ls):
                print(str(preci)+' '+str(np.nanmean(precs))[:5]+' '+str(np.nanstd(precs))[:5])
            all_prec = sum(prec_ls, [])
            print('Average precision: {} {}'.format(str(np.nanmean(all_prec))[:5], str(np.nanstd(all_prec))[:5]))


def prepare_pbir_features(predicted_features_ext, gt_labels_ext, resume_dir, itq=False):
    query_index = []
    for ll in np.unique(gt_labels_ext):
        query_index.append(np.where(gt_labels_ext == ll)[0][0:10])

    query_index = np.concatenate(query_index)

    query_index_bool = np.zeros(gt_labels_ext.shape[0]).astype(bool)
    query_index_bool[query_index] = True

    predicted_features_query = predicted_features_ext[query_index_bool]
    gt_labels_query = gt_labels_ext[query_index_bool]
    predicted_features_gallery = predicted_features_ext[np.logical_not(query_index_bool)]
    gt_labels_gallery = gt_labels_ext[np.logical_not(query_index_bool)]

    if itq:
        q_predicted_features_gallery, q_predicted_features_query = \
            compressITQ(predicted_features_gallery, predicted_features_query)
        scores = - cdist(q_predicted_features_query, q_predicted_features_gallery, metric='hamming')
        print('hamming distance calculated')
    else:
        scores = - cdist(predicted_features_query, predicted_features_gallery)
        print('euclidean distance calculated')

    with open(os.path.join(resume_dir, 'features_photo.pickle'),'wb') as fh:
        pickle.dump([predicted_features_gallery, gt_labels_gallery,
                     predicted_features_query, gt_labels_query,
                     None],fh)

    return predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores


def prepare_sbsr_features(predicted_features_ext, gt_labels_ext, resume_dir, itq=False):

    query_index = []
    for ll in np.unique(gt_labels_ext):
        query_index.append(np.where(gt_labels_ext==ll)[0][0:10])

    query_index = np.concatenate(query_index)

    query_index_bool = np.zeros(gt_labels_ext.shape[0]).astype(bool)
    query_index_bool[query_index] = True

    predicted_features_query = predicted_features_ext[query_index_bool]
    gt_labels_query = gt_labels_ext[query_index_bool]
    predicted_features_gallery = predicted_features_ext[np.logical_not(query_index_bool)]
    gt_labels_gallery = gt_labels_ext[np.logical_not(query_index_bool)]

    if itq:
        q_predicted_features_gallery, q_predicted_features_query = \
            compressITQ(predicted_features_gallery, predicted_features_query)
        scores = - cdist(q_predicted_features_query, q_predicted_features_gallery, metric='hamming')
        print('hamming distance calculated')
    else:
        scores = - cdist(predicted_features_query, predicted_features_gallery)
        print('euclidean distance calculated')

    with open(os.path.join(resume_dir, 'features_sketch.pickle'),'wb') as fh:
        pickle.dump([predicted_features_gallery, gt_labels_gallery,
                     predicted_features_query, gt_labels_query,
                     None], fh)

    return predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores


def prepare_features(args, model=None, itq=False):

    transformations = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize([224,224]),
        # transforms.Normalize(immean, imstd)
    ])

    if args.dataset == 'tuberlin':
        args.num_classes = 220
        test_zero_ext = TUBerlinDataset(split='zero', version='ImageResized_ready', zero_version = args.zero_version, \
                                            transform=transformations, aug=False, first_n_debug=1000000)
        test_zero = TUBerlinDataset(split='zero', zero_version = args.zero_version, transform=transformations, aug=False,
                                        first_n_debug=1000000)

    elif args.dataset == 'sketchy':
        if args.zero_version == 'zeroshot2':
            args.num_classes = 104
        else:
            args.zero_version = 'zeroshot1'
            args.num_classes = 100
        test_zero_ext = SketchyDataset(split='zero', version='all_photo', zero_version=args.zero_version, \
                                          transform=transformations, aug=False)
        test_zero = SketchyDataset(split='zero', zero_version=args.zero_version, transform=transformations, aug=False)

    else:
        print('not support dataset', args.dataset)

    datasets = [test_zero.dataset, test_zero.labels, test_zero_ext.dataset, test_zero_ext.labels]

    zero_loader_ext = DataLoader(dataset=test_zero_ext,
                                 batch_size=args.batch_size, shuffle=False, num_workers=8)

    zero_loader = DataLoader(dataset=test_zero, batch_size=args.batch_size, shuffle=False, num_workers=8)

    print(str(datetime.datetime.now()) + ' data loaded.')

    if model is None:
        model = HashingModel(args.arch, args.num_hashing, args.num_classes)
        model = nn.DataParallel(model).to(device)
        print(str(datetime.datetime.now()) + ' model inited.')

        # resume from a checkpoint
        if args.resume_file:
            resume = os.path.join(args.resume_dir, args.resume_file)
        else:
            resume = os.path.join(args.resume_dir, 'checkpoint.pth.tar')

        resume_from_checkpoint(model, resume)

    cudnn.benchmark = True

    predicted_features_gallery, gt_labels_gallery = get_features(zero_loader_ext, model, args.pretrained)

    predicted_features_query, gt_labels_query = get_features(zero_loader, model, args.pretrained, 0)

    if itq:
        q_predicted_features_gallery, q_predicted_features_query = \
            compressITQ(predicted_features_gallery, predicted_features_query)
        scores = - cdist(q_predicted_features_query, q_predicted_features_gallery, metric='hamming')
        print('hamming distance calculated')
    else:
        scores = - cdist(predicted_features_query, predicted_features_gallery)
        print('euclidean distance calculated')

    with open(os.path.join(args.resume_dir, 'features_zero.pickle'),'wb') as fh:
        pickle.dump([predicted_features_gallery, gt_labels_gallery,
                     predicted_features_query, gt_labels_query,
                     None],fh)

    return predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, datasets


@torch.no_grad()
def get_features(data_loader, model, pretrained=False, tag=1):
    # switch to evaluate mode
    model.eval()
    features_all = []
    targets_all = []
    for i, (input, target) in enumerate(data_loader):
        if i % 10==0:
            print(i, end=' ', flush=True)

        tag_input = (torch.ones(input.size()[0],1)*tag).to(device)
        input = input.to(device)

        # compute output
        features = model.module.original_model.features(input, tag_input)
        if pretrained:
            features = model.module.original_model.avg_pool(features)
            features = features.view(features.size(0), -1)#[:, ::16]
        else:
            features = model.module.original_model.hashing(features)

        features = F.normalize(features)
        features = features.cpu().detach().numpy()

        features_all.append(features.reshape(input.size()[0],-1))
        targets_all.append(target.detach().numpy())

    print('')

    features_all = np.concatenate(features_all)
    targets_all = np.concatenate(targets_all)

    print('Features ready: {}, {}'.format(features_all.shape, targets_all.shape))

    return features_all, targets_all


def eval_AP_inner(inst_id, scores, gt_labels, top=None):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    tot_pos = np.sum(pos_flag)

    sort_idx = np.argsort(-scores)
    tp = pos_flag[sort_idx]
    fp = np.logical_not(tp)

    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        prec = tp / (tp + fp)
    except:
        print(inst_id, tot_pos)
        return np.nan

    ap = VOCap(rec, prec)
    return ap


def VOCap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre)-2,-1,-1):
        mpre[ii] = max(mpre[ii], mpre[ii+1])

    msk = [i!=j for i,j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk]-mrec[0:-1][msk])*mpre[1:][msk])
    return ap


def eval_precision(inst_id, scores, gt_labels, top=100):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]

    top = min(top, tot)

    sort_idx = np.argsort(-scores)
    return np.sum(pos_flag[sort_idx][:top])/top


if __name__ == '__main__':
    # args = parser.parse_args()
    args = get_train_args()
    # main(args)
    if args.dataset == 'sketchy2':
        args.dataset = 'sketchy'
        args.zero_version = 'zeroshot2'

    sys.stdout = Logger(os.path.join(args.resume_dir, 'log_test.txt'))

    evaluate(args, args.resume_dir, get_precision=args.precision, recompute=args.recompute, visualize=args.visualize)
