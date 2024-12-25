from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np 
from collections import OrderedDict


class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):    
        return input.round()
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_lv, size):
        ctx.save_for_backward(torch.Tensor([n_lv, size]))
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        saved, = ctx.saved_tensors
        n_lv, size = int(saved[0]), float(saved[1])

        if n_lv == 0:
            return grad_output, None, None
        else:
            scale = 1 / np.sqrt(n_lv * size)
            return grad_output.mul(scale), None, None

class Q_Act(nn.Module):
    def __init__(self):
        super(Q_Act, self).__init__()
        self.n_lv       = 0
        self.s          = Parameter(torch.ones(1))
        

    def initialize(self, n_lv, tensor, per_token=False):
        self.n_lv = n_lv
        self.qmin = 0
        self.qmax = n_lv - 1
        self.per_token = per_token
        if per_token:
            x = tensor.permute(0,2,1) # b,d,l
            x = x.reshape(-1, x.shape[-1]) # bd, l
            max_val = x.max(dim=0, keepdim=True)[0] # 1, l
            min_val = x.min(dim=0, keepdim=True)[0] # 1, l
            val = (max_val - min_val) / self.qmax
            del self.s
            self.register_parameter("s",torch.nn.Parameter(val.unsqueeze(2)))
            self.z = -(min_val.unsqueeze(2) / self.s).round()
        else:
            max_val = tensor.max()
            min_val = tensor.min()
            
            val = (max_val - min_val) / self.qmax
            self.s.data = torch.tensor(val) 
            self.z = -(min_val / self.s).round()
        print(f'Q_Act initialized {self.name} : {self.s.max()}')
    
    
    def forward(self, x):
        if self.n_lv == 0:
            return x
        else:
            # tmp = x[:,98,:].clone()
            # if self.per_token:
            #     s = (x.abs().max(dim=-1, keepdim=True)[0] / (self.n_lv // 2 - 1)).clamp(1e-5)
            # else:
            if self.per_token:
                # max_val = x.max(dim=-1, keepdim=True)[0]
                # min_val = x.min(dim=-1, keepdim=True)[0]
                # s = (max_val - min_val) / self.qmax
                # z = -(min_val / s).round()
                s = self.s
                z = self.z
            else:
                s = self.s
                z = self.z
            x = F.hardtanh(x / s + z, self.qmin, self.qmax)
            x = RoundQuant.apply(x - z) * s
            # x[:,98,:] = tmp
            return x

class Q_Linear(nn.Linear):
    def __init__(self, *args, act_func=None, **kargs):
        super(Q_Linear, self).__init__(*args, **kargs)
        self.act_func = act_func
        self.n_lv = 0
        # self.s = Parameter(torch.Tensor(1))
        self.s = Parameter(torch.Tensor(self.weight.shape[0],1))
        
    def initialize(self, n_lv, per_channel=False):
        self.n_lv = n_lv
        if per_channel:
            max_val = self.weight.data.abs().max(dim=1, keepdim=True)[0]
            self.s.data = (max_val / (self.n_lv // 2 - 1))
        else:
            del self.s
            max_val = self.weight.abs().max()
            val = max_val / (self.n_lv // 2 - 1)
            self.register_parameter("s",torch.nn.Parameter(val))
            
        print(f'Q_Linear max value initialized {self.name} : {self.s.max()}')
        print("==============================================")


    def _weight_quant(self):
        s = self.s
        weight = F.hardtanh(self.weight / s,- (self.n_lv // 2 - 1), self.n_lv // 2 - 1)
        weight = RoundQuant.apply(weight) * s
        return weight
    

    def _weight_int(self):
        with torch.no_grad():        
            weight = F.hardtanh(self.weight / self.s,- (self.n_lv // 2 - 1), self.n_lv // 2 - 1)
            weight = RoundQuant.apply(weight)
        return weight
        
        
    def forward(self, x):
        if self.act_func is not None:
            x = self.act_func(x)

        if self.n_lv == 0:
            return F.linear(x, self.weight, self.bias)
        else:
            try:
                weight = self._weight_quant()
                if weight.dtype != x.dtype:
                    x.half()
                    return F.linear(x.half(), weight, self.bias)            
                else:
                    return F.linear(x, weight, self.bias)            
            except:
                breakpoint()


def initialize(model, loader, n_lvw, n_lva, act=False, weight=False, per_channel=False, per_token=False):
    model.cuda()
    def initialize_hook(module, input, output):
        if isinstance(module, (Q_Act)) and act:
            module.initialize(n_lva, input[0], per_token)

        if isinstance(module, (Q_Linear)) and weight:
            module.initialize(n_lvw, per_channel)


    hooks = []

    for name, module in model.named_modules():
        hook = module.register_forward_hook(initialize_hook)
        hooks.append(hook)

    
    model.eval()    
    # model.cpu()
    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                output = model.module(input)
            else:
                output = model(input)
        break

    # model.cuda()
    for hook in hooks:
        hook.remove()

class QuantOps(object):
    initialize = initialize
    Act = Q_Act
    Linear = Q_Linear
