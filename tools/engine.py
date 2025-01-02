# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

import timm
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from tools import utils
import statistics
from tqdm import tqdm
import time

@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def time_measure(data_loader, model, amp_autocast, test_turn):
    model.eval()
    B = data_loader.batch_size
    C, H, W = data_loader.dataset[0][0].shape
    sample_image = torch.randn(B,C,H,W).to('cuda')

    time_per_batch = []
    for i in tqdm(range(test_turn)):
        with amp_autocast():
            start = time.time()
            output = model(sample_image)
            time_per_batch.append(time.time() - start)
    print(f"{statistics.median(time_per_batch):.8f} sec")
    return statistics.median(time_per_batch)
