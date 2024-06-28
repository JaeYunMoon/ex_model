import copy 
import math 

from munch import Munch 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class StyleEncoder(nn.Module):
    """
    input = image, label
        image : style을 빼오고자 하는 이미지 
        label : index역할을 하는 int형 

    output = (batch,style_dim)
    """
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size # 256이미지 크기 기준 64 
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s
    

class segtoConst(nn.Module):
    def __init__(self,img_size=256,style_dim=64,num_domain=2,max_conv_dim=512)
        super().__init__()

    """
    input = segimage
    output = 4,4,512 
    """