import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from utils.common import *
from torchvision import models as tv
from torch.nn.parameter import Parameter
import os


class MultiScaleVGGPerceptualLoss(torch.nn.Module):
    def __init__(self, num_scales=3, lam=1, lam_p=1):
        super(MultiScaleVGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss()
        self.lam = lam
        self.lam_p = lam_p
        self.num_scales = num_scales

    def forward(self, outputs, gt1, feature_layers=[2]):
        outputs = outputs[:self.num_scales]
        loss = 0.0
        for i in range(len(outputs)):
            out = outputs[i]
            gt = F.interpolate(gt1, scale_factor=1/(2**i), mode='bilinear', align_corners=False)
            loss += self.lam_p * self.loss_fn(out, gt, feature_layers=feature_layers) + self.lam * F.l1_loss(out, gt)

        return loss


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
