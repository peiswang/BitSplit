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

from quant import ofwa, ofwa_rr, ofwa_rr_dw

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', metavar='DIR', default='./data',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenet_v2_quan',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
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
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

if torch.cuda.is_available():
    quan_device = torch.device("cuda:0")
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        pretrained_device = torch.device("cuda:1")
    else:
        pretrained_device = torch.device("cuda:0")
else:
    quan_device = torch.device("cpu")
    pretrained_device = torch.device("cpu")


conv_pretrained_modules = []
conv_quant_modules = []
act_quant_modules = []
global feat, prev_feat, conv_feat
def hook(module, input, output):
    global feat
    feat = output.data.cpu().numpy()
def current_input_hook(module, inputdata, outputdata):
    global prev_feat
    prev_feat = inputdata[0].data#.cpu()#.numpy()
def conv_hook(module, inputdata, outputdata):
    global conv_feat
    conv_feat = outputdata.data#.cpu()#.numpy()


def main():
    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    model_quant = models.__dict__[args.arch]()
    model_pretrained = models.__dict__[args.arch]()
    assert(args.pretrained is not '')
    model_quant.load_state_dict(torch.load(args.pretrained))
    model_pretrained.load_state_dict(torch.load(args.pretrained))

    model_quant = model_quant.cuda()
    model_pretrained = model_pretrained.cuda(pretrained_device)

    # collecting conv layers (last fc layer 8bit)
    for m in model_pretrained.modules():
        if isinstance(m, nn.Conv2d):
            conv_pretrained_modules.append(m)
    for m in model_quant.modules():
        if isinstance(m, nn.Conv2d):
            conv_quant_modules.append(m)

    # collecting activation quantization layers (last fc layer 8bit)
    for m in model_quant.modules():
        if isinstance(m, Quantization):
            m.set_bitwidth(args.act_bit_width)
            act_quant_modules.append(m)
    if len(act_quant_modules) > 0:
        act_quant_modules[-1].set_bitwidth(8)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

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

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        ############################
        if args.scales:
            print("activation ('{}-bit')...".format(args.act_bit_width))
            scales = np.load(args.scales)
            # enable feature map quantization
            for index, q_module in enumerate(act_quant_modules):
                q_module.set_scale(scales[index])
        else:
            print('no activation scales given, use FP32')

        print('validate quantization...')
        validate(val_loader, model_quant, criterion, args)
        return

    # set to eval mode
    model_pretrained.eval()
    model_quant.eval()

    args.prefix = args.arch + '/A' + str(args.act_bit_width) + 'W' + str(args.weight_bit_width)
    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)
    # print(args)


    ############################
    #### Weight Quantization
    ############################
    for (idx, conv, conv_quant) in zip(range(len(conv_quant_modules)), conv_pretrained_modules, conv_quant_modules):
        prefix = args.prefix + '/conv{}'.format(idx)
        kernel_size = conv.kernel_size[0]
        if idx==0:
            conduct_ofwa(train_loader, model_pretrained, model_quant, conv, conv_quant, None, 8, prefix=prefix, ec=False)
        elif kernel_size == 1:
            conduct_ofwa(train_loader, model_pretrained, model_quant, conv, conv_quant, None, args.weight_bit_width, prefix=prefix, ec=False)
        elif kernel_size == 3:
            conduct_ofwa(train_loader, model_pretrained, model_quant, conv, conv_quant, None, args.weight_bit_width, prefix=prefix, dw=True, ec=False)
        else:
            assert(False)

    ## quantize last fc
    conv = model_pretrained.classifier[-1]
    conv_quant = model_quant.classifier[-1]
    conduct_ofwa(train_loader, model_pretrained, model_quant, conv, conv_quant, None, 8, prefix=args.prefix+'/fc', ec=False)



    ## Load Weight Quantization
    for (idx, conv, conv_quant) in zip(range(len(conv_quant_modules)), conv_pretrained_modules, conv_quant_modules):
        prefix = args.prefix + '/conv{}'.format(idx)
        kernel_size = conv.kernel_size[0]
        if idx==0:
            load_ofwa(conv, conv_quant, None, 8, prefix=prefix)
        else:
            load_ofwa(conv, conv_quant, None, args.weight_bit_width, prefix=prefix)

    ## quantize last fc
    conv = model_pretrained.classifier[-1]
    conv_quant = model_quant.classifier[-1]
    load_ofwa(conv, conv_quant, None, 8, prefix=args.prefix+'/fc')


    ############################
    #### Activation Quantization
    ############################
    print("quantizing ('{}-bit')...".format(args.act_bit_width))
    update(train_loader, model_quant, criterion, args, 200)
    if args.scales:
        print('given scales of activations')
        scales = np.load(args.scales)
        # enable feature map quantization
        for index, q_module in enumerate(act_quant_modules):
            q_module.set_scale(scales[index])
    else:
        print('quantizing activations')
        quantize(train_dataset, model_quant, args)

    validate(val_loader, model_quant, criterion, args)
    print('update ...')
    update(train_loader, model_quant, criterion, args, 200)
    print('validate quantization...')
    validate(val_loader, model_quant, criterion, args)

    save_state_dict(model_quant.state_dict(), args.prefix, filename='state_dict.pth')


def quantize(train_dataset, model, args):

    def get_safelen(x):
        x = x / 10
        y = 1
        while(x>=10):
            x = x / 10
            y = y * 10
        return int(y)

    act_sta_len = 3000000
    feat_buf = np.zeros(act_sta_len)

    scales = np.zeros(len(act_quant_modules))

    print('act quantization modules: ', len(act_quant_modules))

    with torch.no_grad():
        for index, q_module in enumerate(act_quant_modules):
            batch_iterator = iter(torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                             num_workers=args.workers, pin_memory=True))
            images, targets = next(batch_iterator)
            images = images.cuda()
            targets = targets.cuda()

            #### ADD HANDLE ####
            handle = q_module.register_forward_hook(hook)

            model(images)

            #global feat
            feat_len = feat.size
            per_batch = min(get_safelen(feat_len), 100000)
            n_batches = int(act_sta_len / per_batch)

            failed = True
            while(failed):
                failed = False
                print('Extracting features for ', n_batches, ' batches...')
                for batch_idx in range(0, n_batches):
                    images, targets = next(batch_iterator)
                    images = images.cuda(device=quan_device, non_blocking=True)
                    # forward
                    model(images)

                    #global feat
                    if q_module.signed:
                        feat_tmp = np.abs(feat).reshape(-1)
                    else:
                        feat_tmp = feat[feat>0].reshape(-1)
                        if feat_tmp.size < per_batch:
                            per_batch = int(per_batch / 10)
                            n_batches = int(n_batches * 10)
                            failed = True
                            break
                    np.random.shuffle(feat_tmp)
                    feat_buf[batch_idx*per_batch:(batch_idx+1)*per_batch] = feat_tmp[0:per_batch]

                if(not failed):
                    print('Init quantization... ')
                    scales[index] = q_module.init_quantization(feat_buf)
                    print(scales[index])
                    np.save(os.path.join(args.prefix, 'act_' + str(args.act_bit_width) + '_scales.npy'), scales)
            #### REMOVE HANDLE ####
            handle.remove()

    np.save(os.path.join(args.prefix, 'act_' + str(args.act_bit_width) + '_scales.npy'), scales)
    # enable feature map quantization
    for index, q_module in enumerate(act_quant_modules):
        q_module.set_scale(scales[index])


def conduct_ofwa_act(train_loader, model_pretrained, model_quant, conv, conv_quant, q_module, act_bitwidth, prefix=None, ec=False):
    if q_module is None:
        return
    beta = act_quan_one_layer(train_loader, model_quant, q_module)
    with open(prefix + '_beta.pkl', 'wb') as f:
        pickle.dump({'beta': beta}, f, pickle.HIGHEST_PROTOCOL)
    if ec:
        conv_quant.weight.data.copy_(torch.from_numpy(W_r))
        if q_module is not None:
            q_module.set_scale(beta)

def load_ofwa_act(conv, conv_quant, q_module, act_bitwidth, prefix=None):
    if q_module is not None:
        with open(prefix + '_beta.pkl', 'rb') as f:
            B_alpha = pickle.load(f)
            beta = B_alpha['beta']
            q_module.set_scale(beta)


def conduct_ofwa(train_loader, model_pretrained, model_quant, conv, conv_quant, q_module, bitwidth, prefix=None, dw=False, ec=False):
    # for fc
    if not hasattr(conv, 'kernel_size'):
        W = conv.weight.data#.cpu()
        W_shape = W.shape
        B_sav, B, alpha = ofwa(W.cpu().numpy(), bitwidth)
        with open(prefix + '_fwa.pkl', 'wb') as f:
            pickle.dump({'B': B, 'alpha': alpha}, f, pickle.HIGHEST_PROTOCOL)
        if ec:
            W_r = np.multiply(B, np.expand_dims(alpha, 1)).reshape(W_shape)
            conv_quant.weight.data.copy_(torch.from_numpy(W_r))
        return

    # conv parameters
    kernel_h, kernel_w = conv.kernel_size
    pad_h, pad_w = conv.padding
    stride_h, stride_w = conv.stride

    handle_prev = conv_quant.register_forward_hook(current_input_hook)
    handle_conv = conv.register_forward_hook(conv_hook)

    batch_iterator = iter(train_loader)

    # weights and bias
    W = conv.weight.data#.cpu()
    if conv.bias is None:
        bias = torch.zeros(W.shape[0]).to(conv.weight.device)
    else:
        bias = conv.bias.data#.cpu()
    print(W.shape)

    # feat extract
    n_batches = 30
    per_batch = 400
    input, target = next(batch_iterator)
    input_pretrained = input.cuda(device=pretrained_device, non_blocking=True)
    input_quan = input.cuda(device=quan_device, non_blocking=True)
    model_pretrained(input_pretrained)
    model_quant(input_quan)
    # print(prev_feat.shape)
    # print(conv_feat.shape)
    [prev_feat_n, prev_feat_c, prev_feat_h, prev_feat_w] = prev_feat.shape
    [conv_feat_n, conv_feat_c, conv_feat_h, conv_feat_w] = conv_feat.shape

    X = torch.zeros(n_batches*per_batch, prev_feat_c, kernel_h, kernel_w).to(quan_device)
    Y = torch.zeros(n_batches*per_batch, conv_feat_c).to(pretrained_device)
    print(X.shape)
    print(Y.shape)

    for batch_idx in range(0, n_batches):
        input, target = next(batch_iterator)
        input_pretrained = input.cuda(device=pretrained_device, non_blocking=True)
        model_pretrained(input_pretrained)
        input_quan = input.cuda(device=quan_device, non_blocking=True)
        model_quant(input_quan)
    
        prev_feat_pad = torch.zeros(prev_feat_n, prev_feat_c, prev_feat_h+2*pad_h, prev_feat_w+2*pad_w).to(quan_device)
        prev_feat_pad[:, :, pad_h:pad_h+prev_feat_h, pad_w:pad_w+prev_feat_w] = prev_feat
        prev_feat_pad = prev_feat_pad.unfold(2, kernel_h, stride_h).unfold(3, kernel_w, stride_w).permute(0,2,3,1,4,5)
        [feat_pad_n, feat_pad_h, feat_pad_w, feat_pad_c, feat_pad_hh, feat_pad_ww] = prev_feat_pad.shape
        assert(feat_pad_hh==kernel_h)
        assert(feat_pad_ww==kernel_w)
        # prev_feat_pad = prev_feat_pad.reshape(feat_pad_n*feat_pad_h*feat_pad_w, -1)
        prev_feat_pad = prev_feat_pad.reshape(feat_pad_n*feat_pad_h*feat_pad_w, feat_pad_c, kernel_h, kernel_w)
        rand_index = list(range(prev_feat_pad.shape[0]))
        random.shuffle(rand_index)
        rand_index = rand_index[0:per_batch]
        X[per_batch*batch_idx:per_batch*(batch_idx+1),:] = prev_feat_pad[rand_index, :]

        conv_feat_tmp = conv_feat.permute(0,2,3,1).reshape(-1, conv_feat_c) - bias
        Y[per_batch*batch_idx:per_batch*(batch_idx+1),:] = conv_feat_tmp[rand_index, :]

    handle_prev.remove()
    handle_conv.remove()


    ## ofwa init
    W_shape = W.shape

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    W = W.reshape(W_shape[0], -1)
    B_sav, B, alpha = ofwa(W.cpu().numpy(), bitwidth)
    if dw:
        B, alpha = ofwa_rr_dw(X, Y, B_sav, alpha, bitwidth, max_epoch=100)
    else:
        B, alpha = ofwa_rr(X, Y, B_sav, alpha, bitwidth, max_epoch=100)
    with open(prefix + '_rr_b30x400_e100.pkl', 'wb') as f:
        pickle.dump({'B': B, 'alpha': alpha}, f, pickle.HIGHEST_PROTOCOL)


def load_ofwa(conv, conv_quant, q_module, bitwidth, prefix=None):
    # for fc
    if not hasattr(conv, 'kernel_size'):
        W = conv.weight.data#.cpu()
        W_shape = W.shape
        with open(prefix + '_fwa.pkl', 'rb') as f:
            B_alpha = pickle.load(f)
            B = B_alpha['B']
            alpha = B_alpha['alpha']
        W_r = np.multiply(B, np.expand_dims(alpha, 1)).reshape(W_shape)
        conv_quant.weight.data.copy_(torch.from_numpy(W_r))
        return

    # weights and bias
    W = conv.weight.data#.cpu()
    W_shape = W.shape

    with open(prefix + '_rr_b30x400_e100.pkl', 'rb') as f:
        B_alpha = pickle.load(f)
        B = B_alpha['B']
        alpha = B_alpha['alpha']
    W_r = np.multiply(B, np.expand_dims(alpha, 1)).reshape(W_shape)
    conv_quant.weight.data.copy_(torch.from_numpy(W_r))

def act_quan_one_layer(train_loader, model, q_module):

    def get_safelen(x):
        x = x / 10
        y = 1
        while(x>=10):
            x = x / 10
            y = y * 10
        return int(y)

    act_sta_len = 3000000
    feat_buf = np.zeros(act_sta_len)

    with torch.no_grad():
        batch_iterator = iter(train_loader)
        images, targets = next(batch_iterator)
        images = images.cuda(device=quan_device)

        #### ADD HANDLE ####
        handle = q_module.register_forward_hook(hook)

        model(images)

        #global feat
        feat_len = feat.size
        per_batch = min(get_safelen(feat_len), 100000)
        n_batches = int(act_sta_len / per_batch)

        failed = True
        while(failed):
            failed = False
            print('Extracting features for ', n_batches, ' batches...')
            for batch_idx in range(0, n_batches):
                images, targets = next(batch_iterator)
                images = images.cuda(device=quan_device, non_blocking=True)
                # forward
                model(images)

                #global feat
                if q_module.signed:
                    feat_tmp = np.abs(feat).reshape(-1)
                else:
                    feat_tmp = feat[feat>0].reshape(-1)
                    if feat_tmp.size < per_batch:
                        per_batch = int(per_batch / 10)
                        n_batches = int(n_batches * 10)
                        failed = True
                        break
                np.random.shuffle(feat_tmp)
                feat_buf[batch_idx*per_batch:(batch_idx+1)*per_batch] = feat_tmp[0:per_batch]

            if(not failed):
                print('Init quantization... ')
                scale = q_module.init_quantization(feat_buf)
                print(scale)
        #### REMOVE HANDLE ####
        handle.remove()

    return scale


def update(train_loader, model, criterion, args, max_iter):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    global_iter = 0
    flag = True
    end = time.time()
    with torch.no_grad():
        while(flag):
            for i, (input, target) in enumerate(train_loader):
                global_iter = global_iter + 1
                # measure data loading time
                data_time.update(time.time() - end)

                #if args.gpu is not None:
                #    input = input.cuda(args.gpu, non_blocking=True)
                #target = target.cuda(args.gpu, non_blocking=True)
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

                # compute gradient and do SGD step
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           0, i, len(train_loader), batch_time=batch_time,
                           data_time=data_time, loss=losses, top1=top1, top5=top5))
                if global_iter == max_iter:
                    flag = False
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
            #if args.gpu is not None:
            #    input = input.cuda(args.gpu, non_blocking=True)
            #target = target.cuda(args.gpu, non_blocking=True)
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
