import os
import random
import json
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from metrics import dice

criterion_ce = torch.nn.CrossEntropyLoss()    # combine softmax and CE

def get_num(path):
    """labelxxxx.nii.gz"""
    return os.path.basename(path).split('.')[0][-4:]

def tup_to_dict(file_list):
    out_list = []
    for tup in file_list:
        dct = {}
        dct['img_file'] = tup[0]
        dct['image'] = tup[0]
        dct['label'] = tup[1]
        dct['edge'] =  tup[2]
        out_list.append(dct)
    return out_list

def mfb_ce(input, target):
    """
    median frequency balancing weighted cross-entropy loss
    :param input: B * C * H * W * D
    :param target:  B * H * W * D
    :return:
    """
    # self.class_mapping = {
    #     "1": "spleen",  #
    #     "2": 'left kidney',  # 3 -> 2
    #     "3": 'gallbladder',  # 4 -> 3
    #     "4": 'esophagus',  # 5 -> 4
    #     "5": 'liver',  # 6 -> 5
    #     "6": 'stomach',  # 7 -> 6
    #     "7": 'pancreas',  # 11 -> 7
    #     "8": 'duodenum',  # 14 -> 8
    # }
    mfb_weights = torch.Tensor([0.01296055, 0.6061528, 1., 6.39558407, 10.95443216,
                                0.09695645, 0.41963412, 2.04366128, 1.85810754]).cuda()
    # softmax + ce
    return F.cross_entropy(input, target, mfb_weights, reduction='mean')

def bce2d_new(input, target, reduction='mean'):
    """EGNet ICCV 2019"""
    assert(input.size() == target.size())
    # for every positions, return 1 if same
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()
    # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg
    # sigmoid + ce
    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

class AvgMeter(object):
    """
    Acc meter class, use the update to add the current acc
    and self.avg to get the avg acc
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)

def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # (N, C, D, H, W) -> (C, N * D * H * W)
    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))

def compute_dsc(predicted_map, target, no_class):
    """average DSC scores across subjects in a batch"""
    # to one-hot
    predicted_map = expand_as_one_hot(predicted_map.squeeze(1).long(), no_class).cpu().numpy()
    target = expand_as_one_hot(target.squeeze(1).long(), no_class).cpu().numpy()
    # DSC
    organs_dsc = np.zeros((target.shape[0], target.shape[1]-1))
    for b in range(target.shape[0]):  # batch_size
        for i in range(1, no_class):
            tmp_dsc = dice(predicted_map[b, i, ...], target[b, i, ...], nan_for_nonexisting=False)
            organs_dsc[b, i-1] = tmp_dsc
    return np.average(organs_dsc, axis=0)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

def get_files_from_txt(txt_files):
    """
    :return: [(img_file, lab_file), ...]
    """
    with open(txt_files, 'r') as f:
        this_files = f.readlines()
    files = [(i.rstrip('\n').split(',')[0], i.rstrip('\n').split(',')[1]) for i in this_files]
    return files

def do_split(txt_file):
    """save cv fold json"""
    files = get_files_from_txt(txt_file)
    kf = KFold(n_splits=4, random_state=123, shuffle=True)
    train, val = {}, {}

    for i, (train_index, val_index) in enumerate(kf.split(files)):
        tmp_tr = [files[idx] for idx in train_index]
        tmp_val = [files[idx] for idx in val_index]
        train[f'fold_{i}']  = tmp_tr
        val[f'fold_{i}'] = tmp_val

    obj = {'train': train, 'val': val}
    with open(txt_file.replace('all_high_resolution.txt', 'cv_high_resolution.json'), 'w') as f:
        json.dump(obj, f, indent=4)

def get_fold_from_json(json_file, fold):
    """read json to get training and validation files list"""
    with open(json_file, 'r') as f:
        a = json.load(f)
    return a['train'][f'fold_{fold}'], a['val'][f'fold_{fold}']

def save_volume(case, vol, out_fd):
    """
    :param case: files path
    :param vol: (np.ndarray) H * W * D
    :return:
    """
    os.makedirs(out_fd, exist_ok=True)
    out_name = '_'.join([case.split('/')[-3], case.split('/')[-1]])
    out_name = out_name.replace('img', 'pseg')
    affine = nib.load(case).affine
    nib_vol = nib.Nifti1Image(vol.astype(np.int32), affine)
    nib.save(nib_vol, os.path.join(out_fd, out_name))

def save_edge(case, edge, out_fd):
    """
    :param case:
    :param edge: (np.ndarray)
    :param out_fd:
    :return:
    """
    os.makedirs(out_fd, exist_ok=True)
    out_name = '_'.join([case.split('/')[-3], case.split('/')[-1]])
    out_name = out_name.replace('img', 'edge')
    out_name_np = out_name.replace('.nii.gz', '.npz')

    affine = nib.load(case).affine
    edge_lt = (edge * 255.).astype(np.int32)

    nib_vol = nib.Nifti1Image(edge_lt, affine)
    nib.save(nib_vol, os.path.join(out_fd, out_name))
    # np.save(os.path.join(out_fd, out_name_np), edge)

if __name__ == '__main__':
    do_split('/data/yzf/dataset/Project/ranet-dataset/all_high_resolution.txt')