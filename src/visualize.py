from __future__ import absolute_import
from __future__ import print_function

# __all__ = ['visualize_ranked_results']

import numpy as np
import torch
import os
import os.path as osp
import shutil
import cv2
import math
import pdb
# from matplotlib import pyplot as plt

from utils import mkdir_if_missing


GRID_SPACING = 10
QUERY_EXTRA_SPACING = 30 # 90
BW = 5 # border width
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
FRONT = cv2.FONT_HERSHEY_SIMPLEX


def visualize_ranked_results(distmat, dataset, width=224, height=224, save_dir='', topk=10):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
    saved in folders each containing a tracklet.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))

    # query, gallery = dataset
    query = dataset[0]
    gallery = dataset[2]
    qpids = dataset[1]
    gpids = dataset[3]

    indices = np.argsort(-distmat, axis=1)

    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, (tuple, list)):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched else 'FALSE'
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3)) + '_' + suffix
            else:
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)


    for q_idx in range(num_q):
        qimg_path = query[q_idx]
        qpid = qpids[q_idx]
        qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path

        qimg = cv2.imread(qimg_path)
        qimg = cv2.resize(qimg, (width, height))
        qimg = cv2.copyMakeBorder(qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # resize twice to ensure that the border width is consistent across images
        qimg = cv2.resize(qimg, (width, height))
        num_cols = topk + 1
        # grid_img = 255 * np.ones((height, num_cols*width+topk*GRID_SPACING+QUERY_EXTRA_SPACING, 3), dtype=np.uint8)
        grid_img = 255 * np.ones((math.ceil((topk/10)) * height + (topk//10) * GRID_SPACING,
                                  11*width+10*GRID_SPACING+QUERY_EXTRA_SPACING, 3), dtype=np.uint8)
        grid_img[:height, :width, :] = qimg

        rank_idx = 1
        for g_idx in indices[q_idx,:]:

            gimg_path = gallery[g_idx]
            gpid = gpids[g_idx]
            invalid = False
            # invalid = (qcamid == gcamid)

            if not invalid:
                matched = gpid==qpid
                gimg = cv2.imread(gimg_path)
                gimg = cv2.resize(gimg, (width, height))

                border_color = GREEN if matched else RED
                gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
                gimg = cv2.resize(gimg, (width, height))
                start = ((rank_idx - 1) % 10 + 1)*(width + GRID_SPACING) + QUERY_EXTRA_SPACING
                end = start + width
                hstart = (math.ceil(rank_idx/10)-1) * (height + GRID_SPACING)
                hend = hstart + height

                try:
                    grid_img[hstart:hend, start: end, :] = gimg
                except:
                    pdb.set_trace()
            rank_idx += 1
            if rank_idx > topk:
                break

        imname = osp.basename(osp.splitext(qimg_path_name)[0])

        mkdir_if_missing(os.path.join(save_dir, str(qpid)))
        cv2.imwrite(osp.join(save_dir, str(qpid), imname+'.jpg'), grid_img)

        if (q_idx+1) % 100 == 0:
            print('- done {}/{}'.format(q_idx+1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))

