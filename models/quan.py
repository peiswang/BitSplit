import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantization(nn.Module):
    def __init__(self, islinear=False, bit_width=8):
        super(Quantization, self).__init__()
        # self.scale = None
        self.in_scale = None
        self.out_scale = None
        self.signed = islinear
        self.bit_width = bit_width
        self.set_bitwidth(self.bit_width)

    def set_bitwidth(self, bit_width):
        self.bit_width = bit_width
        if self.signed:
            self.max_val = (1 << (self.bit_width - 1)) - 1
            self.min_val = - self.max_val
        else:
            self.max_val = (1 << self.bit_width) - 1
            self.min_val = 0

    def set_scale(self, scale):
        self.set_inscale(scale)
        self.set_outscale(scale)

    def set_inscale(self, in_scale):
        self.in_scale = in_scale
        if isinstance(self.in_scale, (float, np.float32, np.float64)):
            pass
        else:
            self.in_scale = torch.tensor(self.in_scale).view(1, -1, 1, 1)

    def set_outscale(self, out_scale):
        self.out_scale = out_scale
        if isinstance(self.out_scale, (float, np.float32, np.float64)):
            pass
        else:
            self.out_scale = torch.tensor(self.out_scale).view(1, -1, 1, 1)

    def init_quantization(self, x):
        print("x.max", np.max(x))
        print("x.min", np.min(x))
        print("max_val", self.max_val)
        assert(np.min(x)>=0)
    
        circle_detection_queue = [0,]*5
    
        alpha = np.max(np.fabs(x)) / self.max_val
        alpha_old = alpha * 0
        n_iter = 0
        circle_detection_queue[n_iter] = alpha
        while(np.sum(alpha!=alpha_old)):
            q = x / alpha
            q = np.clip(np.round(q), self.min_val, self.max_val)
    
            alpha_old = alpha;
            alpha = np.sum(x*q) / np.sum(q*q)
    
            if alpha in circle_detection_queue:
                break
            n_iter += 1
            circle_detection_queue[n_iter%5] = alpha
        return alpha

    def forward(self, x):
        if self.in_scale is None:
            assert(self.out_scale is None)
            return x
        if not isinstance(self.in_scale, (float, np.float32, np.float64)):
            self.in_scale = self.in_scale.to(x.device)
        if not isinstance(self.out_scale, (float, np.float32, np.float64)):
            self.out_scale = self.out_scale.to(x.device)
        return torch.clamp(torch.round(x/self.in_scale), self.min_val, self.max_val) * self.out_scale

