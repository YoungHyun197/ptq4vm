# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from tools.datasets import build_dataset
from tools.engine import evaluate, time_measure
import tools.utils as utils
from tools.samplers import RASampler
from contextlib import suppress
from tools import models_mamba


import torch.nn as nn

def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)  

def get_args_parser():
    parser = argparse.ArgumentParser('PTQ4VM and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    
    # # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--time_compare', action='store_true', help='comparing time Kernel vs FP')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # amp about
    parser.add_argument('--if_amp', action='store_true')
    parser.add_argument('--no_amp', action='store_false', dest='if_amp')
    parser.set_defaults(if_amp=False)
 
    parser.add_argument('--local-rank', default=0, type=int)
    
    # quantization parameters 
    parser.add_argument("--act_scales", default='', help='path for act_scale checkpoint')
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--n-lvw", type=int, default=256)
    parser.add_argument("--n-lva", type=int, default=256)
    parser.add_argument("--qmode", type=str, default='ptq4vm', choices=['ptq4vm'])
    parser.add_argument('--train-batch', default=256, type=int)
    
    parser.add_argument('--lr-a', type=float, default=5e-4, metavar='LR',
                        help='activation stepsize lr (default: 5e-4)')
    parser.add_argument('--lr-w', type=float, default=5e-4, metavar='LR',
                        help='weight stepsize lr (default: 5e-4)')
    parser.add_argument('--lr-s', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    
    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )
                   
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    # amp about
    amp_autocast = suppress
    
    if args.time_compare:
        checkpoint = torch.load(args.resume, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print(msg)

        # time measure for FP
        time_measure(data_loader_val, model, amp_autocast, 100)

        # time measure for Kernel
        from ptq4vm.quantizer import QuantOps as Q

        if Q is not None:
            for name, module in model_without_ddp.named_modules():
                if isinstance(module, nn.Linear):
                    if 'out_proj' in name or 'in_proj' in name or 'x_proj' in name or 'dt_proj' in name:
                        quantlinear = Q.Linear(module.in_features, module.out_features, 
                                            act_func=Q.Act(), bias=False if module.bias is None else True, 
                                            device=module.weight.device)
                        quantlinear.weight.data = module.weight
                        if module.bias is not None:
                            quantlinear.bias.data = module.bias
                        add_new_module(name, model_without_ddp, quantlinear)
                        del quantlinear

        for name, module in model_without_ddp.named_modules():
            module.name = name 

        act_scales = torch.load(args.act_scales)
        from ptq4vm.jlss import JLSS
        JLSS(model_without_ddp, args, data_loader_train, device, act_scales)

        for name, module in model_without_ddp.named_modules():
                if isinstance(module, Q.Linear):
                    module.set_real_int8()
                    module.act_func.set_real_int8()
        
        time_measure(data_loader_val, model_without_ddp, amp_autocast, 100)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        
        if args.qmode == "ptq4vm":    
            from ptq4vm.quantizer import QuantOps as Q
        else:
            Q = None
        
        if Q is not None:
            for name, module in model_without_ddp.named_modules():
                if isinstance(module, nn.Linear):
                    if 'out_proj' in name or 'in_proj' in name or 'x_proj' in name or 'dt_proj' in name:
                        quantlinear = Q.Linear(module.in_features, module.out_features, 
                                            act_func=Q.Act(), bias=False if module.bias is None else True, 
                                            device=module.weight.device)
                        quantlinear.weight.data = module.weight
                        if module.bias is not None:
                            quantlinear.bias.data = module.bias
                        add_new_module(name, model_without_ddp, quantlinear)
                        del quantlinear
                        
        for name, module in model_without_ddp.named_modules():
            module.name = name  
              
        if args.qmode == "ptq4vm":  
            act_scales = torch.load(args.act_scales)
            from ptq4vm.jlss import JLSS
            JLSS(model_without_ddp, args, data_loader_train, device, act_scales)

        test_stats = evaluate(data_loader_val, model, device, amp_autocast)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
