import torch
import torch.nn as nn
import torch.nn.functional as F
from src.OctaveConv2 import *


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.ReLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class GFEM(nn.Module):
    """Global Feature Enhancement Module."""

    def __init__(self, c1, c2, final=True):  # ch_in, ch_out, final
        super().__init__()
        self.layer1 = Conv(c1, c1 // 2, 3, 1, d=2)
        self.layer2 = nn.Sequential(Conv(c1 + c1 // 2, c1 // 4, 1, 1), Conv(c1 // 4, c1 // 4, 3, 1, d=4))
        self.layer3 = nn.Sequential(Conv(c1 + c1 // 2 + c1 // 4, c1 // 4, 1, 1), Conv(c1 // 4, c1 // 4, 3, 1, d=8))
        self.final = Conv(c1 + c1 // 2 + c1 // 4 + c1 // 4, c2, 1, 1) if final else nn.Identity()

    def forward(self, x):
        x1 = torch.concat([x, self.layer1(x)], 1)
        x2 = torch.concat([x1, self.layer2(x1)], 1)
        x = torch.concat([x2, self.layer3(x2)], 1)
        return self.final(x)


class SecNet(nn.Module):
    """Semantic Network."""

    def __init__(self, c1, c2, c3):
        super().__init__()
        self.cv1 = Conv(c1, c1, 1, 1)
        self.cv2 = Conv(c3, c3, 1, 1)
        self.lfem = Conv(c1 + c2 + c3, c2, 1, 1)
        self.gfem = GFEM(c2, c2, final=False)

    def forward(self, x):
        x1, x2, x3 = x[0], x[1], x[2]
        h, w = x2.shape[2:]
        x1 = F.interpolate(self.cv1(x1), size=[h, w], mode='bilinear', align_corners=False)
        x3 = F.interpolate(self.cv2(x3), size=[h, w], mode='bilinear', align_corners=False)
        return self.gfem(self.lfem(torch.concat([x1, x2, x3], 1)))


class HFEM(nn.Module):
    """High-frequency Extraction Module."""

    def __init__(self, c1, c2, e=0.5, alpha=0.125):  # ch_in, ch_out, number, expansion, alpha
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  
        self.relu = nn.ReLU()
        self.first = FirstOctaveCBR(self.c, self.c, kernel_size=(3, 3), alpha=alpha, padding=1)
        self.m = OctaveCB(self.c, self.c, kernel_size=(3, 3), alpha=alpha, padding=1)
        self.last = LastOCtaveCBR(self.c, self.c, kernel_size=(3, 3), alpha=alpha, padding=1)

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        x_h_res, x_l_res = self.first(a)
        x_h, x_l = self.m((x_h_res, x_l_res))

        x_h += x_h_res
        x_l += x_l_res

        x_h = self.relu(x_h)
        x_l = self.relu(x_l)

        x = self.last((x_h, x_l))
        return self.cv2(torch.cat(((a + x), b), 1))


class EdgeNet(nn.Module):
    """Edge Network."""

    def __init__(self, c1, c2, c, alpha=0.125):  # ch_in, ch_out, ch_in[1:number], alpha
        super().__init__()
        self.hfem = HFEM(c1, c2, alpha=alpha)
        self.m1 = nn.ModuleList(Conv(x, c2, 1, 1) for x in c)
        self.m2 = nn.ModuleList(nn.Sequential(Conv(2 * c2, c2, 1, 1), HFEM(c2, c2, alpha=alpha)) for _ in c)
        self.cv = nn.Conv2d(c2, 1, kernel_size=1)

    def forward(self, x):
        h, w = x[0].shape[2:]
        edge = self.hfem(x[0])
        for i, (m1, m2) in enumerate(zip(self.m1, self.m2), 1):
            x[i] = F.interpolate(m1(x[i]), size=[h, w], mode='bilinear', align_corners=False)
            edge = m2(torch.concat([edge, x[i]], 1))
        return self.cv(edge)
