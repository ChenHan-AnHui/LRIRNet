import torch
import torch.nn as nn
import torch.nn.functional as F
from src.mobilenet_v3 import mobilenet_v3_small
from src.resnet_d import resnet18_
from src.block import SecNet, EdgeNet


class LRIRNet(nn.Module):
    def __init__(self, encode_num: int = 5,
                 decode_num: int = 4,
                 out_ch: int = 1):
        super().__init__()
        self.encode_modules = nn.ModuleList()
        mobilenet = mobilenet_v3_small()
        self.layer0 = mobilenet.features[:1]
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            mobilenet.features[1:2], mobilenet.features[2:4], mobilenet.features[4:9], mobilenet.features[9:12]
        del mobilenet
        for i in range(encode_num):
            self.encode_modules.append(eval('self.layer' + str(i)))
        self.secnet = SecNet(16, 24, 48)
        self.edgenet = EdgeNet(16, 16, [40, 40])

        self.decode_modules = nn.ModuleList()
        resnet_ = resnet18_()
        self.layer1_, self.layer2_, self.layer3_, self.layer4_ = \
            resnet_.layer1, resnet_.layer2, resnet_.layer3, resnet_.layer4
        del resnet_
        for i in range(decode_num):
            self.decode_modules.append(eval('self.layer' + str(decode_num - i) + '_'))

        self.out_conv = nn.Conv2d(decode_num * 40, out_ch, kernel_size=1)

    def forward(self, x):
        _, _, h, w = x.shape

        # collect encode outputs
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)
        encode_outputs[2] = self.secnet(encode_outputs[1:4])
        edge = encode_outputs[0]

        # collect decode outputs
        x = encode_outputs.pop()
        decode_outputs = []
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
            x = m(torch.concat([x, x2], dim=1))
            decode_outputs.insert(0, x)

        # collect edge output
        edge = self.edgenet([edge, decode_outputs[2], decode_outputs[0]])
        edge = F.interpolate(edge, size=[h, w], mode='bilinear', align_corners=False)
        edge = torch.sigmoid(edge)

        # collect side upsample outputs
        for i in range(len(decode_outputs)):
            decode_outputs[i] = F.interpolate(decode_outputs[i], size=[h, w], mode='bilinear', align_corners=False)

        x = self.out_conv(torch.concat(decode_outputs, dim=1))
        x = x + edge * x

        return x
