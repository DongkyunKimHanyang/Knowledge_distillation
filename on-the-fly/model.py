import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class BottleNeck(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1):
        super(BottleNeck,self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * 4)

        self.residual = nn.Sequential()
        if stride !=1 or in_channel != out_channel * 4:
            self.residual =nn.Sequential(nn.Conv2d(in_channel,out_channel * 4, kernel_size=1, stride=stride,bias=False),
                                         nn.BatchNorm2d(out_channel * 4))

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.residual(x)
        out = F.relu(out)
        return out

class Shared_layers(nn.Module):
    def __init__(self):
        super(Shared_layers,self).__init__()
        self.in_channel = 16
        n = (110-2) // 9

        self.conv1 = nn.Conv2d(3, 16,kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv_2x = self._make_layer(16, n)
        self.conv_3x = self._make_layer(32, n, stride=2)

    def _make_layer(self,out_channel,num_block,stride=1):
        strides = [stride] + [1] * out_channel
        layers = []
        for i in range(num_block):
            layers.append(BottleNeck(self.in_channel,out_channel,strides[i]))
            self.in_channel = 4 * out_channel
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv_2x(out)
        out = self.conv_3x(out)
        return out

class branch(nn.Module):
    def __init__(self,num_classes):
        super(branch,self).__init__()
        self.in_channel = 128
        n = (110 - 2) // 9

        self.conv_4x = self._make_layer(64,n,2)
        self.avg_pool = nn.AvgPool2d(8)
        self.output_proj = nn.Linear(64 * 4, num_classes)

    def _make_layer(self,out_channel,num_block, stride):
        strides = [stride] + [1] * out_channel
        layers = []
        for i in range(num_block):
            layers.append(BottleNeck(self.in_channel, out_channel, strides[i]))
            self.in_channel = 4 * out_channel
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv_4x(x)
        out = self.avg_pool(out)
        out = out.view(out.size(0),-1)
        out = self.output_proj(out)
        return out

class ONE_model(nn.Module):
    def __init__(self,num_classes):
        super(ONE_model,self).__init__()

        self.shared_layers = Shared_layers()

        self.gate_avgpool = nn.AvgPool2d(16)
        self.gate_dense = nn.Linear(self.shared_layers.in_channel, 3)
        self.gate_bn = nn.BatchNorm1d(3)

        self.branches = nn.ModuleList(branch(num_classes) for i in range(3))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        shared_out = self.shared_layers(x)

        gate_out = self.gate_avgpool(shared_out)
        gate_out = gate_out.view(gate_out.size(0), -1)
        gate_out = self.gate_dense(gate_out)
        gate_out = self.gate_bn(gate_out)
        gate_out = F.relu(gate_out)
        gate_out = F.softmax(gate_out,1)

        branch_logits = torch.cat([self.branches[i](shared_out)[:,None,...] for i in range(3)], dim = 1)
        pred_logits = torch.mul(gate_out[...,None],branch_logits).sum(1)
        return branch_logits, pred_logits


class Resnet_110_model(nn.Module):
    def __init__(self,num_classes):
        super(Resnet_110_model,self).__init__()

        self.shared_layers = Shared_layers()
        self.last_block = branch(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        shared_out = self.shared_layers(x)
        pred_logits = self.last_block(shared_out)
        return None, pred_logits
