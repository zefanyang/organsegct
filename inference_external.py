#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 6/25/2021 10:41 AM
# @Author: yzf
"""External Validation on the WORD dataset"""
import sys
import json
import argparse
import random
import time
import torch
import logging
import pandas as pd
import SimpleITK as sitk
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cacheio.Dataset import Compose, RegularDataset, LoadImage, Clip, ForeNormalize, ToTensor
from utils import tup_to_dict, expand_as_one_hot, get_fold_from_json
from visualizers.batch_visualizer import *
from metrics import dice, hausdorff_distance_95, avg_surface_distance_symmetric

from models.unet_nine_layers.unet_l9_deep_sup import UNetL9DeepSup
from models.unet_nine_layers.unet_l9_deep_sup_full_scheme import UNetL9DeepSupFullScheme

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='3')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--net', type=str, default='unet_l9_ds_full_scheme', choices=['unet_l9_ds', 'unet_l9_ds_full_scheme'])
parser.add_argument('--init_channels', type=int, default=16)
parser.add_argument('--num_class', type=int, default=9, choices=[9])
parser.add_argument('--organs', type=list,
                    default=['bg', 'spleen', 'left kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'pancreas', 'duodenum'])
parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training.')
parser.add_argument('--val_json', type=str, default='/data/yzf/dataset/organct/external/namesval.json')
parser.add_argument('--basepath', type=str, default='/data/yzf/dataset/organct/external/preprocessedval')
parser.add_argument('--checkpointfd', type=str, required=True, help='output folder that stores model checkpoints')
parser.add_argument('--bestckp', action='store_true', default=False, help='indicator showing whether or not to use the best checkpoint')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.bestckp:
    args.ckp_file = os.path.join(args.checkpointfd, 'model_best.pth.tar')
else:
    args.ckp_file = os.path.join(args.checkpointfd, 'checkpoint.pth.tar')

def get_model(args):
    model = None
    if args.net == 'unet_l9_ds':
        model = UNetL9DeepSup(1, args.num_class, init_ch=args.init_channels)
    elif args.net == 'unet_l9_ds_full_scheme':
        model = UNetL9DeepSupFullScheme(1, args.num_class, init_ch=args.init_channels)

    if model is None:
        raise ValueError('Model is None.')
    return model

def parse_data(data):
    img_file = data['img_file']
    image = data['image']
    label = data['label']
    return img_file, image, label

def get_dataloader(args):
    f = open(args.val_json)
    nameslist = json.load(f)
    f.close()
    val_list = [(os.path.join(args.basepath, 'image', _), os.path.join(args.basepath, 'label', _)) for _ in nameslist]
    d_val_list = []
    for tup in val_list:
        dct = {}
        dct['img_file'] = tup[0]
        dct['image'] = tup[0]
        dct['label'] = tup[1]
        d_val_list.append(dct)

    val_transforms = Compose([LoadImage(keys=['image', 'label']),
                              Clip(keys=['image'], min=-250., max=200.),
                              ForeNormalize(keys=['image'], mask_key='label'),
                              ToTensor(keys=['image', 'label'])])

    # Regular Dataset
    val_dataset = RegularDataset(data=d_val_list, transform=val_transforms)

    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)
    return val_dataloader

def inference(args):
    model = get_model(args).cuda()
    model.load_state_dict(torch.load(args.ckp_file)['state_dict'])
    model.eval()

    dsc_df, hd95_df, assd_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    dsc_ls, hd95_ls, assd_ls, index_ls = [], [], [], []

    val_dataloader = get_dataloader(args)
    for i, val_data in enumerate(val_dataloader):
        tic = time.time()
        case, volume, seg = parse_data(val_data)
        spacing = sitk.ReadImage(case).GetSpacing()
        volume = volume.cuda()
        seg = seg.cuda()
        with torch.no_grad():
            if 'edge' in args.net:
                seg_score, _ = model(volume)
            # elif 'rfp' in args.net or 'cascaded' in args.net:
            #     _, seg_score = model(volume)
            elif 'full_scheme' in args.net:
                _, seg_score, _ = model(volume)
            else:
                seg_score = model(volume)
        seg_probs = torch.softmax(seg_score, dim=1)
        seg_map = torch.argmax(seg_probs, dim=1, keepdim=True)

        case_dsc, case_hd95, case_assd = [], [], []
        tgt_oh = expand_as_one_hot(seg.squeeze(1).long(), args.num_class).cpu().numpy()  # target
        prd_oh = expand_as_one_hot(seg_map.squeeze(1).long(), args.num_class).cpu().numpy()  # prediction
        for cls in range(1, args.num_class):
            dsc = dice(test=prd_oh[:, cls, ...], reference=tgt_oh[:, cls, ...], nan_for_nonexisting=True)
            hd95 = hausdorff_distance_95(test=prd_oh[:, cls, ...], reference=tgt_oh[:, cls, ...], voxel_spacing=spacing, nan_for_nonexisting=True)
            assd = avg_surface_distance_symmetric(test=prd_oh[:, cls, ...], reference=tgt_oh[:, cls, ...], voxel_spacing=spacing, nan_for_nonexisting=True)

            case_dsc.append(dsc)
            case_hd95.append(hd95)
            case_assd.append(assd)

        # simple_idx = '_'.join([case[0].split('/')[-3][:4], case[0].split('/')[-1][3:7]])
        simple_idx = os.path.basename(case[0])
        index_ls.append(simple_idx)
        dsc_ls.append(case_dsc)
        hd95_ls.append(case_hd95)
        assd_ls.append(case_assd)

        logging.info('Finish evaluating {}. Take {:.2f} s, DSC: {}, Avg DSC: {:.4f}'
                     .format(simple_idx, time.time() - tic, [round(_, 4) for _ in case_dsc], np.mean(case_dsc)))

    dsc_df = pd.DataFrame(dsc_ls, columns=args.organs[1:], index=index_ls)
    hd95_df = pd.DataFrame(hd95_ls, columns=args.organs[1:], index=index_ls)
    assd_df = pd.DataFrame(assd_ls, columns=args.organs[1:], index=index_ls)

    dsc_df.to_csv(os.path.join(args.out_fd, 'dsc.csv'))
    hd95_df.to_csv(os.path.join(args.out_fd, 'hd95.csv'))
    assd_df.to_csv(os.path.join(args.out_fd, 'assd.csv'))

if __name__ == '__main__':
    args.out_fd = f'./results/externalval/{os.path.basename(args.checkpointfd)}'
    os.makedirs(os.path.join(args.out_fd, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args.out_fd, 'snapshots'), exist_ok=True)

    # logger
    logging.basicConfig(filename=args.out_fd+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # record configurations
    with open(args.out_fd+'/parameters.txt', 'w') as txt:
        for a in vars(args).items():
            txt.write(str(a)+'\n')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    inference(args)