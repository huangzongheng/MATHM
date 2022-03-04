import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from skimage.transform import warp, AffineTransform
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


random_flip = transforms.RandomHorizontalFlip()
random_rot = transforms.RandomRotation(20)
random_affine = transforms.RandomAffine(degrees=(-30, 30), translate=(0.0, 0.0), scale=(0.7, 1.3))


# def scatch_aug(img):
#     # _, h, w = img.shape
#     h, w = img.size
#     h0, w0 = img.size
#
#     # img = random_flip(img)
#     # img = random_rot(img)
#     # img = random_affine(img)
#
#     # mask = F.pad(mask, (80, )*4)    # 先pad再做transform，防止图像越界
#     mask = transforms.Pad((80, )*4, fill=255)(img)
#     mask = random_affine(mask)
#     mask = transforms.ToTensor()(mask)
#     mask = 1 - mask[:1]   # h w
#
#
#     eft_range = []
#     for i in range(2):
#         x_mask = (mask.mean(i+1) > 1e-6).squeeze()
#         x_range = torch.arange(x_mask.shape[0])[x_mask]
#         eft_range.append((x_range.min(), x_range.max()))
#     eft_range = torch.tensor(eft_range)     # ymin,ymax;xmin,xmax
#     # eft_range = eft_range / torch.tensor(mask.shape[1:])[:, None].float()
#     # if (torch.tensor((h, w)) - sketch_size).min() < 0:
#     #     mask = transforms.functional.resize(mask, )
#
#     # sx = min(eft_range[0, 0], mask.shape[1] - eft_range[0, 1])
#     # sy = min(eft_range[1, 0], mask.shape[2] - eft_range[1, 1])
#     start = eft_range[:, 0]
#     sketch_size = eft_range[:, 1] - eft_range[:, 0]
#     ty = torch.randint(max(0, h - sketch_size[0])+1, (1,))
#     tx = torch.randint(max(0, w - sketch_size[1])+1, (1,))
#     h = max(h, sketch_size[0])
#     w = max(w, sketch_size[1])
#     mask = F.pad(mask, (tx, mask.shape[1] - start[0]-h, ty, mask.shape[2] - start[1]-w))      # pad 以向右平移
#     mask = mask[:, start[0]:start[0]+h, start[1]:start[1]+w]
#     mask = transforms.functional.resize(mask, (h0, w0))
#
#     mask = 1 - mask
#
#     img = mask.expand(3, -1, -1)
#     # print(img.shape, mask.shape)
#
#     return img


def scatch_aug(img):
    _, h, w = img.shape

    mask = 1 - img[:1]   # h w
    mask = F.pad(mask, (80, ) * 4)    # pad before transforms to avoid generating incomplete sketch
    mask = random_affine(mask)

    eft_range = []
    for i in range(2):
        x_mask = (mask.mean(i + 1) > 1e-6).squeeze()
        x_range = torch.arange(x_mask.shape[0])[x_mask]
        eft_range.append((x_range.min(), x_range.max()))
    eft_range = torch.tensor(eft_range)     # ymin,ymax;xmin,xmax

    start = eft_range[:, 0]
    sketch_size = eft_range[:, 1] - eft_range[:, 0]
    ty = torch.randint(max(0, h - sketch_size[0]) + 1, (1,))
    tx = torch.randint(max(0, w - sketch_size[1]) + 1, (1,))
    h = max(h, sketch_size[0])
    w = max(w, sketch_size[1])
    mask = F.pad(mask, (tx, mask.shape[1] - start[0] - h, ty, mask.shape[2] - start[1] - w))      # shift right
    mask = mask[:, start[0]:start[0] + h, start[1]:start[1] + w]
    mask = transforms.functional.resize(mask, img.shape[1:])
    mask = 1 - mask

    img = mask.expand(3, -1, -1)

    return img


def random_transform(img, scale=0.3, rotate=30, trainslate=10):
    if np.random.random() < 0.3:
        return img

    img = random_flip(img)
    # img = F.pad(img)

    if np.random.random() < 0.5:
        sx = np.random.uniform(1 - scale, 1 + scale)
        sy = np.random.uniform(1 - scale, 1 + scale)
    else:
        sx = 1.0
        sy = 1.0

    if np.random.random() < 0.5:
        rx = np.random.uniform(-rotate * 2.0 * np.pi / 360.0, +rotate * 2.0 * np.pi / 360.0)
    else:
        rx = 0.0

    if np.random.random() < 0.5:
        tx = np.random.uniform(-trainslate, trainslate)
        ty = np.random.uniform(-trainslate, trainslate)
    else:
        tx = 0.0
        ty = 0.0
    img_aug = transforms.functional.affine(img, angle=rx, translate=(tx, ty), scale=sx, shear=(0, 0))

    return img_aug
