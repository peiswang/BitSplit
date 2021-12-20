import argparse
import os
import random
import shutil
import time
import warnings
import sys
import pickle
import math
import numpy as np
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models as models
from models.quan import Quantization

from quant import ofwa, ofwa_rr

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', metavar='DIR', default='./data',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_quan',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=640, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, 
                    help='use pre-trained model')
parser.add_argument('--weight-bit-width', default=4, type=int,
                    help='activation quantization bit-width')
parser.add_argument('--act-bit-width', default=8, type=int,
                    help='activation quantization bit-width')
parser.add_argument('--scales', default='', type=str, metavar='PATH',
                    help='path to the pre-calculated scales (.npy file)')


def main():
    args = parser.parse_args()

    # create model
    model = models.__dict__[args.arch]()
    assert(args.pretrained is not '')
    model.load_state_dict(torch.load(args.pretrained))
    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

    if 'alexnet' in args.arch:
        input_size = 227
    else:
        input_size = 224

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    args.epoch_size = len(train_dataset) // args.batch_size

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    update_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    start = time.time()

    args.prefix = args.arch + '/A' + str(args.act_bit_width) + 'W' + str(args.weight_bit_width)
    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)
    # print(args)

    # collecting activation quantization layers (last fc layer 8bit)
    act_quant_modules = []
    for m in model.modules():
        if isinstance(m, Quantization):
            m.set_bitwidth(args.act_bit_width)
            act_quant_modules.append(m)
    act_quant_modules[-1].set_bitwidth(8)

    ###### for evaluation ######
    if args.evaluate:
        if args.scales:
            print("activation ('{}-bit')...".format(args.act_bit_width))
            scales = np.load(args.scales)
            # enable feature map quantization
            for index, q_module in enumerate(act_quant_modules):
                q_module.set_scale(scales[index])
        else:
            print('no activation scales given, use FP32')

        print('validate quantization...')
        validate(val_loader, model, criterion, args)
        return
    ###### for evaluation ######

    # set to eval mode
    model.eval()

    ############################
    #### Weight Quantization
    ############################
    # Add feature extraction hook
    handle_list = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handle = m.register_forward_hook(_conv_hook_)
            handle_list.append(handle)

    # forward once for feature extraction
    batch_iterator = iter(train_loader)
    images, targets = next(batch_iterator)
    images = images.cuda()
    with torch.no_grad():
        model(images)

    # Remove all handles
    for handle in handle_list:
        handle.remove()

    print("weight quantizing ('{}-bit')...".format(args.weight_bit_width))
    # stem convoution
    conv = model.conv1
    prefix = args.prefix + '/conv1'
    print('\t', prefix)
    conduct_ofwa(conv, 8, prefix=prefix)

    #### quantize 4 blocks
    for layer_idx in range(1, 5):
        current_layer_quan = eval('model.layer{}'.format(layer_idx))
        for block_idx in range(len(current_layer_quan)):
            current_block_quan = current_layer_quan[block_idx]

            # conv1
            conv = current_block_quan.conv1
            prefix = args.prefix + '/layer' + str(layer_idx) + '_block' + str(block_idx) + '_conv1'
            print('\t', prefix)
            conduct_ofwa(conv, args.weight_bit_width, prefix=prefix)

            # conv2
            conv = current_block_quan.conv2
            prefix = args.prefix + '/layer' + str(layer_idx) + '_block' + str(block_idx) + '_conv2'
            print('\t', prefix)
            conduct_ofwa(conv, args.weight_bit_width, prefix=prefix)

            # downsample
            if current_block_quan.downsample is not None:
                conv = current_block_quan.downsample[0]
                prefix = args.prefix + '/layer' + str(layer_idx) + '_block' + str(block_idx) + '_downsample'
                print('\t', prefix)
                conduct_ofwa(conv, args.weight_bit_width, prefix=prefix)

    ## quantize last fc
    conv = model.fc
    q_module = model.quan
    prefix = args.prefix + '/fc'
    print('\t', prefix)
    conduct_ofwa(conv, 8, prefix=prefix)

    print('update ...')
    update(update_loader, model, 200)

    ############################
    #### Activation Quantization
    ############################
    # Add feature extraction hook
    handle_list = []
    for m in model.modules():
        if isinstance(m, Quantization):
            handle = m.register_forward_hook(_quant_hook_)
            handle_list.append(handle)

    # forward once for feature extraction
    batch_iterator = iter(train_loader)
    images, targets = next(batch_iterator)
    images = images.cuda()
    with torch.no_grad():
        model(images)

    # Remove all handles
    for handle in handle_list:
        handle.remove()

    print("activation quantizing ('{}-bit')...".format(args.act_bit_width))
    conduct_activation_quantization(args, act_quant_modules)

    print('update ...')
    update(update_loader, model, 200)

    end = time.time()

    print('validate quantization...')
    validate(val_loader, model, criterion, args)

    print('Run Time:', end-start, ' s')

    save_state_dict(model.state_dict(), args.prefix, filename='state_dict.pth')


def _conv_hook_(conv, input, output):

    kernel_h, kernel_w = conv.kernel_size
    pad_h, pad_w = conv.padding
    stride_h, stride_w = conv.stride

    # weights and bias
    W = conv.weight.data#.cpu()
    if conv.bias is None:
        bias = torch.zeros(W.shape[0]).to(conv.weight.device)
    else:
        bias = conv.bias.data#.cpu()

    # feat extract
    n_samples = 15000

    prev_feat = input[0].data#.cpu()#.numpy()
    conv_feat = output.data#.cpu()#.numpy()

    [prev_feat_n, prev_feat_c, prev_feat_h, prev_feat_w] = prev_feat.shape
    [conv_feat_n, conv_feat_c, conv_feat_h, conv_feat_w] = conv_feat.shape

    prev_feat_pad = torch.zeros(prev_feat_n, prev_feat_c, prev_feat_h+2*pad_h, prev_feat_w+2*pad_w).cuda() #to(quan_device)
    prev_feat_pad[:, :, pad_h:pad_h+prev_feat_h, pad_w:pad_w+prev_feat_w] = prev_feat
    prev_feat_pad = prev_feat_pad.unfold(2, kernel_h, stride_h).unfold(3, kernel_w, stride_w).permute(0,2,3,1,4,5)
    [feat_pad_n, feat_pad_h, feat_pad_w, feat_pad_c, feat_pad_hh, feat_pad_ww] = prev_feat_pad.shape
    assert(feat_pad_hh==kernel_h)
    assert(feat_pad_ww==kernel_w)
    prev_feat_pad = prev_feat_pad.reshape(feat_pad_n*feat_pad_h*feat_pad_w, feat_pad_c, kernel_h, kernel_w)
    rand_index = list(range(prev_feat_pad.shape[0]))
    random.shuffle(rand_index)
    rand_index = rand_index[0:n_samples]
    X = prev_feat_pad[rand_index]

    conv_feat_tmp = conv_feat.permute(0,2,3,1).reshape(-1, conv_feat_c) - bias
    Y = conv_feat_tmp[rand_index]

    conv.X = X.cpu().numpy()
    conv.Y = Y.cpu().numpy()

def _quant_hook_(q_module, input, output):
    act_sta_len = 200000
    feat = output.data.cpu().numpy()
    if q_module.signed:
        feat_tmp = np.abs(feat).reshape(-1)
    else:
        feat_tmp = feat[feat>0].reshape(-1)

    if feat_tmp.size < act_sta_len:
        q_module.feat_buf = feat_tmp
    else:
        np.random.shuffle(feat_tmp)
        q_module.feat_buf = feat_tmp[0:act_sta_len]


def conduct_activation_quantization(args, act_quant_modules):
    scales = np.zeros(len(act_quant_modules))
    for index, q_module in enumerate(act_quant_modules):
        scales[index] = q_module.init_quantization(q_module.feat_buf)
        # print(scales[index])
    np.save(os.path.join(args.prefix, 'act_' + str(args.act_bit_width) + '_scales.npy'), scales)

    # enable activation quantization
    for index, q_module in enumerate(act_quant_modules):
        q_module.set_scale(scales[index])

def conduct_ofwa(conv, bitwidth, prefix=None):
    # for fc
    if not hasattr(conv, 'kernel_size'):
        W = conv.weight.data#.cpu()
        W_shape = W.shape
        B_sav, B, alpha = ofwa(W.cpu().numpy(), bitwidth)
        with open(prefix + '_fwa.pkl', 'wb') as f:
            pickle.dump({'B': B, 'alpha': alpha}, f, pickle.HIGHEST_PROTOCOL)

    else:
        # weights and bias
        W = conv.weight.data#.cpu()
        if conv.bias is None:
            bias = torch.zeros(W.shape[0]).to(conv.weight.device)
        else:
            bias = conv.bias.data#.cpu()
        # print(W.shape)

        ## ofwa init
        W_shape = W.shape

        W = W.reshape(W_shape[0], -1)
        B_sav, B, alpha = ofwa(W.cpu().numpy(), bitwidth)
        B, alpha = ofwa_rr(conv.X, conv.Y, B_sav, alpha, bitwidth, max_epoch=100)
        with open(prefix + '_rr_s15000_e100.pkl', 'wb') as f:
            pickle.dump({'B': B, 'alpha': alpha}, f, pickle.HIGHEST_PROTOCOL)
    #######################
    # replace fp32 weights with quantized weights
    #######################
    W_r = np.multiply(B, np.expand_dims(alpha, 1)).reshape(W_shape)
    conv.weight.data.copy_(torch.from_numpy(W_r))


def update(update_loader, model, max_iter):
    # switch to train mode
    model.train()
    with torch.no_grad():
        for i, (input, target) in enumerate(update_loader):
            input = input.cuda(non_blocking=True)
            output = model(input)
            if i == max_iter:
                break

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_state_dict(state_dict, path, filename='state_dict.pth'):
    saved_path = os.path.join(path, filename)
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if '.module.' in key:
            new_state_dict[key.replace('.module.', '.')] = state_dict[key].cpu()
        else:
            new_state_dict[key] = state_dict[key].cpu()
    torch.save(new_state_dict, saved_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
