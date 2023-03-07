#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/1/2021 9:36 PM
# @Author: yzf
"""
Make the following modification: adjust the seg_loss function so that it can calculate multiple outputs.
"""
import argparse
import time
import random
import shutil
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import (
    get_fold_from_json, tup_to_dict, poly_lr, AvgMeter, expand_as_one_hot, compute_per_channel_dice,
    compute_dsc, bce2d_new, save_volume, save_edge
)
from cacheio.Dataset import (
    Compose, PersistentDataset, LoadImage,
    Clip, ForeNormalize, RandFlip, RandRotate, ToTensor
)

from visualizers.batch_visualizer import *

from models.unet_nine_layers.unet_l9_deep_sup_full_scheme import UNetL9DeepSupFullScheme

val_freq = 1.

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--size', type=tuple, default=(160, 160, 64))
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--net', type=str, default='unet_deep_sup_full_scheme')  # TODO
parser.add_argument('--init_channels', type=int, default=16)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--N', type=int, default=-1)
parser.add_argument('--momentum', type=float, default=0.9)  # for SGD
parser.add_argument('--weight_decay', type=float, default=3e-4)
parser.add_argument('--num_class', type=int, default=9)
parser.add_argument('--organs', type=list, default=['spleen', 'left kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'pancreas', 'duodenum'])
parser.add_argument('--num_epoch', type=int, default=400)
parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training.')
parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--beta', type=float, default=1.)  # for DSC
parser.add_argument('--beta2', type=float, default=1.)  # for edge
# parser.add_argument('--root', type=str, default='./output/cascaded_vnet')
# parser.add_argument('--cache_dir', type=str, default='./cache/cascaded_vnet')
parser.add_argument('--cv_json', type=str, default='/raid/yzf/data/abdominal_ct/cv_high_resolution.json')

parser.add_argument('--no_escs', type=int, default=4)

def tr_summary(writer, epoch, c_lr, loss_seg, dsc, loss_edge=None):
    writer.add_scalar('tr_monitor/poly_lr', c_lr, epoch)
    writer.add_scalar('tr_monitor/loss_seg', loss_seg.avg, epoch)
    writer.add_scalar('tr_monitor/dsc', dsc.avg, epoch)
    if loss_edge is not None:
        writer.add_scalar('tr_monitor/loss_edge', loss_edge.avg, epoch)

def val_summary(writer, epoch, loss_seg, organs_dsc, avg_dsc, loss_edge=None):
    writer.add_scalar('val_monitor/loss_seg', loss_seg.avg, epoch)
    writer.add_scalar('val_monitor/avg_dsc', avg_dsc.avg, epoch)
    writer.add_scalar('val_dice/spleen', organs_dsc['spleen'].avg, epoch)  # best sorted based on indices
    writer.add_scalar('val_dice/left_kidney', organs_dsc['left kidney'].avg, epoch)
    writer.add_scalar('val_dice/gallbladder', organs_dsc['gallbladder'].avg, epoch)
    writer.add_scalar('val_dice/esophagus', organs_dsc['esophagus'].avg, epoch)
    writer.add_scalar('val_dice/liver', organs_dsc['liver'].avg, epoch)
    writer.add_scalar('val_dice/stomach', organs_dsc['stomach'].avg, epoch)
    writer.add_scalar('val_dice/pancreas', organs_dsc['pancreas'].avg, epoch)
    writer.add_scalar('val_dice/duodenum', organs_dsc['duodenum'].avg, epoch)
    if loss_edge is not None:
        writer.add_scalar('val_monitor/loss_edge', loss_edge.avg, epoch)

def save_checkpoint(state, is_best, fd):
    filename = os.path.join(fd, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(fd, 'model_best.pth.tar'))

def resume_model(model, fd):
    state = torch.load(os.path.join(fd, 'checkpoint.pth.tar'))
    latest_epoch = state['epoch']
    tol_time = state['tol_time']
    model.load_state_dict(state['state_dict'])
    return model, latest_epoch, tol_time

def get_model(args):
    model = None
    if args.net == 'unet_deep_sup_full_scheme':
        model = UNetL9DeepSupFullScheme(1, args.num_class, init_ch=args.init_channels)

    if model is None:
        raise ValueError('Model is None.')
    return model

def _add_edge_files(files_list):
    new_list = []
    for i in files_list:
        edge_file = i[0].replace('preproc_img', 'edge').replace('img', 'edge')
        tup = (i[0], i[1], edge_file)
        new_list.append(tup)
    return new_list

def get_dataloader(args):
    train_list, val_list = get_fold_from_json(args.cv_json, args.fold)
    t_train_list = _add_edge_files(train_list)
    t_val_list = _add_edge_files(val_list)
    # dict
    d_train_list = tup_to_dict(t_train_list)
    d_val_list = tup_to_dict(t_val_list)

    train_transforms = Compose([LoadImage(keys=['image', 'label', 'edge']),
                                Clip(keys=['image'], min=-250., max=200.),
                                ForeNormalize(keys=['image'], mask_key='label'),
                                RandFlip(keys=['image', 'label', 'edge'], spatial_axis=(0, 1), prob=.5),
                                RandRotate(keys=['image', 'label', 'edge'], interp_order=[1, 0, 0], angle=15.0, prob=.5),
                                ToTensor(keys=['image', 'label', 'edge'])])

    val_transforms = Compose([LoadImage(keys=['image', 'label', 'edge']),
                              Clip(keys=['image'], min=-250., max=200.),
                              ForeNormalize(keys=['image'], mask_key='label'),
                              ToTensor(keys=['image', 'label', 'edge'])])

    # Regular Dataset
    # train_dataset = RegularDataset(data=d_train_list, transform=train_transforms)
    # val_dataset = RegularDataset(data=d_val_list, transform=val_transforms)

    # Persistent Dataset
    train_dataset = PersistentDataset(data=d_train_list, transform=train_transforms, cache_dir=args.cache_dir,)
    val_dataset = PersistentDataset(data=d_val_list, transform=val_transforms, cache_dir=args.cache_dir,)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)
    return train_dataloader, val_dataloader

def parse_data(data):
    img_file = data['img_file']
    image = data['image']
    label = data['label']
    edge = data['edge']
    return img_file, image, label, edge

def adjust_lr(optimizer, epoch, max_epoch, initial_lr=1e-4, N=-1):
    """
    We use "poly" learning rate decay strategy and fix the learning rate at N epoch.

    Args:
        epoch: start from 0
        N: N < 0 means always adopting the decay strategy
    """
    # In main process, epoch starts from 1.
    epoch = epoch-1
    N = N-1

    new_lr = poly_lr(epoch, max_epoch, initial_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    if epoch >= N and not N < 0:
        new_lr = poly_lr(N, max_epoch, initial_lr)  # fixed learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return param_group['lr']

def compute_loss(outputs, seg, seg_one_hot):
    score1, score2 = outputs

    def criterion(score, seg, seg_one_hot):
        loss_ce = torch.nn.CrossEntropyLoss()(score, seg.squeeze(1).long())
        prob = F.softmax(score, dim=1)
        dice = compute_per_channel_dice(prob, seg_one_hot)
        loss_dice = 1. - dice.mean()
        return loss_ce, prob, dice, loss_dice

    loss_ce1, prob1, dice1, loss_dice1 = criterion(score1, seg, seg_one_hot)
    loss_ce2, prob2, dice2, loss_dice2 = criterion(score2, seg, seg_one_hot)

    loss_dice = loss_dice1 + loss_dice2
    loss_ce = loss_ce1 + loss_ce2

    # only outputs the final segmentation map
    predicted_map = torch.argmax(prob2, dim=1, keepdim=True).float()
    return loss_dice, loss_ce, dice2, predicted_map

def compute_edge_loss(edge, edge_score, mode='weighted'):
    if mode == 'vanilla':
        loss_edge = F.binary_cross_entropy_with_logits(edge_score, edge)
    elif mode == 'weighted':
        loss_edge = bce2d_new(edge_score, edge)
    edge_confidence = F.sigmoid(edge_score)
    return loss_edge, edge_confidence

def train_process_edge(epoch, args, net, optimizer, train_dataloader, writer=None):
    """training w edge"""
    tr_start = time.time()  # timing
    c_lr = adjust_lr(optimizer, epoch=epoch, max_epoch=args.num_epoch, initial_lr=args.lr, N=args.N)
    net.train()

    loss_seg_meter = AvgMeter()
    loss_edge_meter = AvgMeter()
    dsc_meter = AvgMeter()

    len_train_batch = len(train_dataloader)
    for i, tr_data in enumerate(train_dataloader):
        # z, ...
        case, volume, seg, edge = parse_data(tr_data)
        seg_one_hot = expand_as_one_hot(seg.squeeze(1).long(), args.num_class).cuda()
        volume = volume.cuda()
        seg = seg.cuda()
        edge = edge.cuda()

        *outputs, edge_score = net(volume)
        loss_dice, loss_ce, seg_dsc, predicted_map = compute_loss(outputs, seg, seg_one_hot)
        loss_edge, edge_confidence = compute_edge_loss(edge, edge_score)
        loss = loss_dice + args.beta * loss_ce + args.beta2 * loss_edge

        loss_seg_meter.update((loss_dice + args.beta * loss_ce).item(), volume.shape[0])
        loss_edge_meter.update(args.beta2 * loss_edge.item(), volume.shape[0])
        dsc_meter.update((seg_dsc[1:].mean()).item(), volume.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('training epoch: {:d}, take: {:.4f} s'.format(epoch, time.time()-tr_start))
    tr_summary(writer, epoch, c_lr, loss_seg_meter, dsc_meter, loss_edge_meter)

def val_process_edge(epoch, args, net, val_dataloader, writer=None):
    """validation w edge"""
    val_time = time.time()
    net.eval()
    # validation
    val_loss_seg_meter = AvgMeter()
    val_loss_edge_meter = AvgMeter()
    val_organs_dsc_meter = dict([(org, AvgMeter()) for org in args.organs])
    val_avg_dsc_meter = AvgMeter()
    len_val = len(val_dataloader)
    with torch.no_grad():
        for i, val_data in enumerate(val_dataloader):
            case, volume, seg, edge = parse_data(val_data)
            seg_one_hot = expand_as_one_hot(seg.squeeze(1).long(), args.num_class).cuda()
            volume = volume.cuda()
            seg = seg.cuda()
            edge = edge.cuda()
            *outputs, edge_score = net(volume)
            val_loss_dice, val_loss_ce, seg_dice, predicted_map = compute_loss(outputs, seg, seg_one_hot)
            val_loss_edge, val_edge_confidence = compute_edge_loss(edge, edge_score)
            organs_dsc = compute_dsc(predicted_map, seg, args.num_class)

            val_loss_seg_meter.update((val_loss_dice+args.beta*val_loss_ce).item(), volume.shape[0])
            val_loss_edge_meter.update((args.beta2*val_loss_edge).item(), volume.shape[0])
            for ind, key in enumerate(val_organs_dsc_meter.keys()):
                # if not math.isnan(organs_dsc[ind]):
                val_organs_dsc_meter[key].update(organs_dsc[ind].item(), volume.shape[0])
            val_avg_dsc_meter.update(organs_dsc.mean(), volume.shape[0])

            if epoch % args.num_epoch == 0:
                # snapshot
                img = volume[0, 0].cpu().numpy()
                seg = seg[0, 0].cpu().numpy()
                edge = edge[0, 0].cpu().numpy()
                seg_map = predicted_map[0, 0].cpu().numpy()
                edge_prob = val_edge_confidence[0, 0].cpu().numpy()

                v_images = []
                h, w, d = img.shape
                for ind in range(d):
                    im = np.rot90(img[..., ind])
                    se = np.rot90(seg[..., ind])
                    ed = np.rot90(edge[..., ind])
                    se_mp = np.rot90(seg_map[..., ind])
                    ed_pr = np.rot90(edge_prob[..., ind])

                    im = (norm_score(im) * 255.).astype(np.uint8)
                    imRGB = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB);
                    getScoreMap = lambda x, rang: cv2.addWeighted(get_score_map(x, rang),
                                                            0.8, imRGB, 0.2, 0)

                    se = getScoreMap(se, (0, 8))
                    ed = (norm_score(ed) * 255.).astype(np.uint8)
                    se_mp = getScoreMap(se_mp, (0, 8))
                    ed_pr = getScoreMap(ed_pr, None)

                    # add text in images. Occur bugs; im.copy() solves the problem.
                    im = imtext(im.copy(), text='{:d} {:.2f}'.format(ind, organs_dsc.mean()*100),
                                space=(3, 10), color=(255,)*3, thickness=2, fontScale=.6)

                    h_images = [im, se, se_mp, ed, ed_pr]
                    v_images.append(imhstack(h_images, height=160))
                v_images = imvstack(v_images)

                name = Path(case[0]).name
                fd = os.path.join(args.root, 'snapshots')
                os.makedirs(fd, exist_ok=True)
                imwrite(os.path.join(fd, name.replace('.nii.gz', '.jpg')), v_images)

                # save volume
                save_volume(case[0], seg_map, os.path.join(args.root, f'predictions'))
                save_edge(case[0], edge_prob, os.path.join(args.root, f'predictions/edge'))

    print('validation epoch: {:d}, take: {:.4f} s'.format(epoch, time.time() - val_time))
    val_summary(writer, epoch, val_loss_seg_meter, val_organs_dsc_meter, val_avg_dsc_meter, val_loss_edge_meter)

    return val_avg_dsc_meter.avg

def main_worker(args):
    ## child process
    # tensorboard
    start_epoch = 1
    tol_time = 0.
    writer = None
    tbx_root = args.root + '/tbx'
    if not os.path.exists(tbx_root):
        os.makedirs(tbx_root)
    writer = SummaryWriter(log_dir=tbx_root)

    # model training
    best_dsc = .0
    model = get_model(args)
    if args.resume:
        model, latest_epoch, tol_time = resume_model(model, args.root)
        start_epoch = latest_epoch+1

    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_dataloader, val_dataloader = get_dataloader(args)
    for epoch in range(start_epoch, args.num_epoch+1):
        tic = time.time()
        train_process_edge(epoch, args, model, optimizer, train_dataloader, writer)
        if epoch % val_freq == 0:
            dsc = val_process_edge(epoch, args, model, val_dataloader, writer)
        else:
            raise RuntimeError("Need to train with edge")

        tol_time += time.time()-tic
        writer.add_scalar('timing', tol_time/3600., epoch,)

        # remember best dsc and save checkpoint
        is_best = dsc > best_dsc
        best_dsc = max(dsc, best_dsc)
        if is_best:
            with open(args.root + '/record_best.txt', 'w') as f:
                f.write('epoch {0}: {1:.4f}'.format(epoch, best_dsc))
        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'tol_time': tol_time,
            },
            is_best,
            args.root)

    writer.close()

def main():
    args = parser.parse_args()
    # args.cache_dir = f'./cache/{Path(__file__).stem}_fold{args.fold}'
    args.cache_dir = f'./cache/{Path(__file__).stem}_fold{args.fold}_{time.time():.2f}'
    args.root = f'./output/{args.net}_fold{args.fold}'

    # Activate for parameter tuning
    # args.cache_dir = f'./cache/{Path(__file__).stem}_fold{args.fold}_wd{args.weight_decay}'
    # args.root = f'./output/{args.net}_fold{args.fold}_wd{args.weight_decay}'

    # # Activate for ESCs ablation study
    # args.cache_dir = f'./cache/{Path(__file__).stem}_fold{args.fold}_{args.no_escs}_escs'
    # args.root = f'./output/{args.net}_{args.no_escs}_escs_fold{args.fold}'



    ## environment
    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # remove files
    root = Path(args.root)
    cache = Path(args.cache_dir)
    if not args.resume:
        if root.exists():
            # if True:
            if '11' == input("Input '11' to continue removing process: "):
                shutil.rmtree(root)
            else:
                raise RuntimeError("Make sure the former output files are backed up properly")
        if cache.exists():
            shutil.rmtree(cache)  # clear cache
        root_ = root / 'predictions/edge'
        root_.mkdir(parents=True)
        if not root_.is_dir():
            raise ValueError("root must be a directory.")

    # deterministic training for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # log configurations
    with open(root / 'parameters.txt', 'w') as f:
        for a in vars(args).items():
            f.write(str(a)+'\n')

    # main process
    main_worker(args)

if __name__ == '__main__':
    main()