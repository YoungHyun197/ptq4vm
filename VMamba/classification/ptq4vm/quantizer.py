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

class Q_Linear(nn.Linear):
    def __init__(self, *args, act_func=None, **kargs):
        super(Q_Linear, self).__init__(*args, **kargs)
        self.act_func = act_func
        self.n_lv = 0
        self.qmax = self.n_lv // 2 - 1 
        self.qmin = -self.qmax
        self.per_channel = False
        self.s = Parameter(torch.Tensor(1))
        # self.s = Parameter(torch.Tensor(self.weight.shape[0],1))
        self.num = 100
        self.eps = torch.tensor(1e-8)
        self.smoothing = False
        
    def quantize_efficient(self, x_round, scale, zero=0):
        q = torch.clamp(x_round + zero, self.qmin , self.qmax)
        return scale * (q - zero)
    
    def lp_loss(self, pred, tgt, p=2.4):
        x = (pred - tgt).abs().pow(p)
        if not self.per_channel:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)

    def initialize(self, n_lv, per_channel=False, trunc=False):
        x = self.weight * self.act_func.smooth_scale
        
        self.n_lv = n_lv
        self.qmax = n_lv // 2 - 1
        self.qmin = -self.qmax
        self.per_channel = per_channel     
        if not trunc:
            if self.per_channel:
                del self.s
                max_val = x.abs().max(dim=1, keepdim=True)[0]
                val = max_val / self.qmax
                self.register_parameter("s",torch.nn.Parameter(val))
                
            else:
                max_val = x.abs().max()
                self.s.data = max_val / self.qmax
        else:
            
            if self.per_channel:
                x = x.flatten(1)
            else:
                x = x.flatten().unsqueeze(0)

            xmin = x.min(1)[0]
            xmax = x.max(1)[0]
            
            if self.per_channel:
                new_shape = [-1] + [1] * (len(x.shape) -  1)
            
            best_score = torch.zeros_like(xmin) + (1e+10)
            best_min = xmin.clone()
            best_max = xmax.clone()
            
            xrange = torch.max(xmin.abs(), xmax)
            
            for i in range(1, self.num + 1):
                tmp_max = xrange / self.num * i
                scale = torch.max(tmp_max / self.qmax, self.eps)
                if self.per_channel:
                    scale = scale.reshape(new_shape)
                x_round = torch.round(x/scale)
                x_q = self.quantize_efficient(x_round, scale)
                score = self.lp_loss(x, x_q, 2.4)
                best_max = torch.where(score < best_score, tmp_max, best_max)
                best_score = torch.min(score, best_score)
                
            max_val = torch.max(best_max, torch.zeros_like(best_max))
            if self.per_channel:
                del self.s
                val = torch.max(max_val / self.qmax, self.eps).unsqueeze(1)
                self.register_parameter("s",torch.nn.Parameter(val))
            else:
                self.s.data = torch.max(max_val / self.qmax, self.eps)

        self.smoothing = True
        # print("Q_Linear Max s :" +  str(self.s.max()))
 
    def _weight_quant(self): 
        s = self.s 
        if self.smoothing:
            weight = self.weight * self.act_func.smooth_scale
        else:
            weight = self.weight
        weight = F.hardtanh((weight / s), self.qmin, self.qmax)
        weight = RoundQuant.apply(weight) * s
        return weight

    def _weight_int(self):
        with torch.no_grad():        
            weight = F.hardtanh(self.weight / self.s,- (self.n_lv // 2 - 1), self.n_lv // 2 - 1)
            weight = torch.round(weight)
        return weight
        
    def forward(self, x):
        if self.act_func is not None:
            x = self.act_func(x)

        if self.n_lv == 0:    
            if self.smoothing:
                weight = self.weight * self.act_func.smooth_scale
            else:
                weight = self.weight
            return F.linear(x, weight, self.bias)
        else:
            try:
                weight = self._weight_quant()
            except:
                breakpoint()
            return F.linear(x, weight, self.bias)

class Q_Act(nn.Module):
    def __init__(self):
        super(Q_Act, self).__init__()
        # n_lv, qmax, qmin -> refer initialize() function
        self.n_lv = 0 
        self.qmax = 0 
        self.qmin = 0
        self.per_channel = False
        self.s = Parameter(torch.Tensor(1))
        self.num = 100
        self.eps = torch.tensor(1e-8)
        self.smoothing = False

        
    def quantize_efficient(self, x_round, scale, zero=0):
        q = torch.clamp(x_round + zero, self.qmin , self.qmax)
        return scale * (q - zero)
    
    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.per_token:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)

    def initialize(self, n_lv, tensor, per_token=False, trunc=False):
        x = tensor / self.smooth_scale
        self.n_lv = n_lv
        self.qmax = n_lv - 1
        self.qmin = 0
        self.per_token = per_token     
        
        # if self.per_token:
        #     self.smoothing=True
        #     return
        
        # self.s.data = x.abs().max() / self.qmax
        # self.smoothing = True
        # trunc = True
        # self.per_token = True
        if not trunc:
            if self.per_token:
                b,l,d = x.shape
                x = x.permute(0,2,1) # b, d, l
                x = x.reshape(-1, l) # bd, l
                x = x.permute(1,0) # l, bd
                del self.s
                max_val = x.max(dim=1, keepdim=True)[0]
                min_val = x.min(dim=1, keepdim=True)[0]
                val = (max_val - min_val) / self.qmax
                self.register_parameter("s",torch.nn.Parameter(val.unsqueeze(0)))
                self.z = torch.round(-min_val.unsqueeze(0) / self.s)
                
            else:
                max_val = x.max()
                min_val = x.min()
                val = (max_val - min_val) / self.qmax
                self.s.data = torch.tensor(val)
                self.z = torch.round(-min_val / self.s)
        else:
            if self.per_token:
                b,l,d = x.shape
                x = x.permute(0,2,1) # b, d, l
                x = x.reshape(-1, l) # bd, l
                x = x.permute(1,0)
            else:
                x = x.flatten().unsqueeze(0)
            
            xmin = x.min(1)[0]
            xmax = x.max(1)[0]
            
            if self.per_token:
                new_shape = [-1] + [1] * (len(x.shape) -  1)
            
            best_score = torch.zeros_like(xmin) + (1e+10)
            best_min = xmin.clone()
            best_max = xmax.clone()
            
            for i in range(1, self.num + 1):
                alpha = i / self.num
                tmp_min = xmin * (1-alpha) + xmax * alpha
                tmp_max = xmin * alpha + xmax * (1-alpha)         
                
                scale = (tmp_max - tmp_min) / (self.qmax - self.qmin)            
                scale = torch.max((tmp_max - tmp_min) / (self.qmax - self.qmin), self.eps)
                zero = torch.round(-tmp_min / scale) + self.qmin
                
                # Reshape for per-channel quantization if needed
                if self.per_token:
                    scale = scale.reshape(new_shape)
                    zero = zero.reshape(new_shape)

                # Perform quantization with the computed scale and zero point
                x_round = torch.round(x / scale)
                x_q = self.quantize_efficient(x_round, scale, zero)

                # Compute score and update best values
                score = self.lp_loss(x, x_q, 2.4)
                best_min = torch.where(score < best_score, tmp_min, best_min)
                best_max = torch.where(score < best_score, tmp_max, best_max)
                best_score = torch.min(best_score, score)

            # Final scale and zero point calculation
            min_val_neg = torch.min(best_min, torch.zeros_like(best_min))
            max_val_pos = torch.max(best_max, torch.zeros_like(best_max))

            max_val = torch.max(best_max, torch.zeros_like(best_max))
            if self.per_token:
                del self.s
                val = torch.max((max_val_pos - min_val_neg) / (self.qmax - self.qmin), self.eps).unsqueeze(1).unsqueeze(0)
                self.register_parameter("s",torch.nn.Parameter(val))
                self.z = torch.clamp(self.qmin - torch.round(min_val_neg.unsqueeze(1).unsqueeze(0) / self.s), self.qmin, self.qmax)
            else:
                self.s.data = torch.max((max_val_pos - min_val_neg) / (self.qmax - self.qmin), self.eps)
                self.z = torch.clamp(self.qmin - torch.round(min_val_neg / self.s), self.qmin, self.qmax)
        self.smoothing = True
        # print("Q_Act Max s :" +  str(self.s.max())) 

        
    def forward(self, x):

        if self.n_lv == 0:
            if self.smoothing:
                return x / self.smooth_scale
            else:
                return x
        else:
            if self.smoothing:
                # s = self.s
                x = x / self.smooth_scale
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
                
            else:
                s = self.s
                z = self.z
                x = F.hardtanh(x / s + z, self.qmin, self.qmax)
                x = RoundQuant.apply(x - z) * s
            return x

            
def initialize(layer, input, n_lvw, n_lva, act=False, weight=False, per_channel=False, per_token=False, trunc=False):    
    def initialize_hook(module, input, output): 
        if isinstance(module, (Q_Linear)) and weight:
            module.initialize(n_lvw, per_channel=per_channel, trunc=trunc)
        if isinstance(module, (Q_Act)):
            module.initialize(n_lva, input[0], per_token=per_token, trunc=trunc)

            
    hooks = []
    for name, module in layer.named_modules():
        hook = module.register_forward_hook(initialize_hook)
        hooks.append(hook)

    with torch.no_grad():
        input = input.to('cuda')
        # residual = residual.to('cuda')
        if isinstance(layer, nn.DataParallel):
            output = layer.module(input)
        else:
            output = layer(input)
            
    for hook in hooks:
        hook.remove()
        
class QuantOps(object):
    initialize = initialize
    Act = Q_Act
    Linear = Q_Linear
