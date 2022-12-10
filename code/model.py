import torch
from torch import nn
# from senet import legacy_seresnext50_32x4d, legacy_seresnext101_32x4d
# from vision_transformer import vit_base_patch16_384
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np

class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class CrossEntropyLossWithLabelSmoothing(nn.Module):
    def __init__(self, n_dim, ls_=0.9):
        super().__init__()
        self.n_dim = n_dim
        self.ls_ = ls_

    def forward(self, x, target):
        target = F.one_hot(target, self.n_dim).float()
        target *= self.ls_
        target += (1 - self.ls_) / self.n_dim

        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, m=0.5, s=30.0):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.m = m
            
    def forward(self, logits, labels, out_dim=2):
        ms = np.array([self.m]*logits.shape[0])
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss     

class ArcNet(nn.Module):
    def __init__(self, back_bone, n_frames):
        super(ArcNet, self).__init__()
        self.model = timm.create_model(back_bone, num_classes=2, pretrained=True, in_chans=3*n_frames)
        self.feat = nn.Linear(1536, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, 2)
        self.model.classifier = nn.Identity()

    def extract(self, x):
        x = self.model.forward_features(x)
        x = x.mean(dim=1)
        return x

    def forward(self, x):
        x = self.extract(x)
        logits_m = self.metric_classify(self.swish(self.feat(x)))
        return logits_m



class Net(nn.Module):
    def __init__(self, back_bone, n_frames):
        super().__init__()
        self.model = timm.create_model(back_bone, num_classes=2, pretrained=True, in_chans=3*n_frames)

    def forward(self, x):
        logit = self.model(x)
        
        return logit
