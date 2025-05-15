from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import math

from torchvision import models
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import Bottleneck, model_urls
import gc

# from SoftPool import soft_pool2d, SoftPool2d
import torch.nn.functional as F



def calc_similiar_penalty(F_1, F_2):
        """Calculate similiar penalty |W_1^T W_2|."""
        F_1_params = F_1.weight.data.T
        F_2_params = F_2.weight.data.T
        similiar_penalty = torch.mean(
            torch.abs(torch.mm(F_1_params.transpose(0, 1), F_2_params)))
        return similiar_penalty

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # out = identity * a_w * a_h
        out = a_w * a_h
        return out
    

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)  


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1

    

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class DomainDiscriminator(nn.Module):
    r"""Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Parameters:
        - **in_feature** (int): dimension of the input feature
        - **hidden_size** (int): dimension of the hidden features

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int):
        super(DomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.relu2(self.bn2(self.layer2(x)))
        y = self.sigmoid(self.layer3(x))
        return y

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr_mult": 1.}]  

class ClassifierBase(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None):
        super(ClassifierBase, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.backbone(x)
        f = f.view(-1, self.backbone.out_features)
        f = self.bottleneck(f)
        predictions = self.head(f)
        return predictions, f

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params
    
class ResNet(models.ResNet):
    """ResNets without fully connected layer"""

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self._out_features = self.fc.in_features
        del self.fc

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = x.view(-1, self._out_features)
        return x

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features    

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model

def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class causal(nn.Module):
    def __init__(self, image_encoder_dim: int,  bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = 256, head: Optional[nn.Module] = None): #num_classes: int,
        super(causal, self).__init__()
        # self.backbone = resnet50(pretrained=True)
        self.image_encoder_dim = image_encoder_dim
        # self.num_classes = num_classes
        self._features_dim = bottleneck_dim
        # if head is None:
        #     self.head1 = nn.Linear(self._features_dim, num_classes)
        #     self.head2 = nn.Linear(self._features_dim, num_classes)
        # else:
        #     self.head1 = head
        #     self.head2 = head
        self.ca = CoordAtt(image_encoder_dim, image_encoder_dim)
        # self.fuse = ChannelAtt(gate_channels=image_encoder_dim, reduction_ratio=2, pool_types=['avg', 'max'])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        bottleneck_F = nn.Sequential(
            nn.Linear(image_encoder_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        bottleneck_B = nn.Sequential(
            nn.Linear(image_encoder_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        bottleneck_E = nn.Sequential(
            nn.Linear(image_encoder_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.bottleneck = nn.ModuleDict({
            'f': bottleneck_F,
            'b': bottleneck_B,
            'e': bottleneck_E,
        })
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=bottleneck_dim,out_features=bottleneck_dim))
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=bottleneck_dim*345,out_features=bottleneck_dim*324)) #65,64\12,9\345,324
        self.linear3 = nn.Sequential(
            nn.Linear(in_features=bottleneck_dim,out_features=bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim))
        self.linear4 = nn.Sequential(
            nn.Linear(in_features=bottleneck_dim*324,out_features=bottleneck_dim*345)) #64,65

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""#torch.Size([64, 1, 256]) torch.Size([64, 65, 256])
        # f = x
        B, N, C = x.shape
        f = x
        H , W = int(np.sqrt(N)), int(np.sqrt(N))
        f = f.view(B,-1)
        if (H * W == N):
            f = self.linear1(f)
        else:
            f = self.linear2(f)
        f = f.reshape(B,C,H,W)
        weight = self.ca(f)
        # weight = self.fuse(f)
        b = weight * f #s
        e = (1-weight) * f  #d
        f2 = self.avgpool(f).view(-1, self.image_encoder_dim)
        b2 = self.avgpool(b).view(-1, self.image_encoder_dim)
        e2 = self.avgpool(e).view(-1, self.image_encoder_dim)

        # f2 = self.bottleneck['f'](f2)
        # b2 = self.bottleneck['b'](b2)
        # e2 = self.bottleneck['e'](e2)
        b = b.view(B,-1)
        if (H * W == N):
            b = self.linear3(b)
        else:
            b = self.linear4(b)
        # predictions1 = self.head1(f2)
        # predictions2 = self.head2(b2)
        # return predictions1, predictions2, f2, b2, e2
        b = b.view(B,N,C)
        return f2,b2,e2,b

def CalculateMean(features, labels, class_num):
    N = features.size(0)
    C = class_num
    A = features.size(1)

    avg_CxA = torch.zeros(C, A).cuda()
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C).cuda()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1.0

    del onehot
    gc.collect()
    for c in range(class_num):
        c_temp = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        c_temp = torch.sum(c_temp, dim=0)
        avg_CxA[c] = c_temp / Amount_CxA[c]
    return avg_CxA.detach()

def Calculate_CV(features, labels, ave_CxA, class_num):
    N = features.size(0)
    C = class_num
    A = features.size(1)

    var_temp = torch.zeros(C, A, A).cuda()
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C).cuda()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1
    Amount_CxAxA = Amount_CxA.view(C, A, 1).expand(C, A, A)
    del Amount_CxA, onehot
    gc.collect()

    avg_NxCxA = ave_CxA.expand(N, C, A)
    for c in range(C):
        features_by_sort_c = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        avg_by_sort_c = avg_NxCxA[:, c, :].mul(NxCxA_onehot[:, c, :])
        var_temp_c = features_by_sort_c - avg_by_sort_c
        var_temp[c] = torch.mm(var_temp_c.permute(1,0), var_temp_c).div(Amount_CxAxA[c])
    return var_temp.detach()


class Cls_Loss(nn.Module):
    def __init__(self, class_num):
        super(Cls_Loss, self).__init__()
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

    def aug(self, s_mean_matrix, t_mean_matrix, fc, features, y_s, labels_s, t_cv_matrix, Lambda):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]
        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels_s.view(N, 1, 1).expand(N, C, A))

        t_CV_temp = t_cv_matrix[labels_s]

        sigma2 = Lambda * torch.bmm(torch.bmm(NxW_ij - NxW_kj, t_CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul(torch.eye(C).cuda().expand(N, C, C)).sum(2).view(N, C)

        sourceMean_NxA = s_mean_matrix[labels_s]
        targetMean_NxA = t_mean_matrix[labels_s]
        dataMean_NxA = (targetMean_NxA - sourceMean_NxA)
        dataMean_NxAx1 = dataMean_NxA.expand(1, N, A).permute(1, 2, 0)

        del t_CV_temp, sourceMean_NxA, targetMean_NxA, dataMean_NxA
        gc.collect()

        dataW_NxCxA = NxW_ij - NxW_kj
        dataW_x_detaMean_NxCx1 = torch.bmm(dataW_NxCxA, dataMean_NxAx1)
        datW_x_detaMean_NxC = dataW_x_detaMean_NxCx1.view(N, C)

        aug_result = y_s + 0.5 * sigma2 + Lambda * datW_x_detaMean_NxC
        return aug_result

    def forward(self, fc, features_source: torch.Tensor, y_s, labels_source, Lambda, mean_source, mean_target, covariance_target):
        aug_y = self.aug(mean_source, mean_target, fc, features_source, y_s, labels_source, covariance_target, Lambda)
        loss = self.cross_entropy(aug_y, labels_source)
        return loss
    
def MI(outputs_target):
    batch_size = outputs_target.size(0)
    softmax_outs_t = nn.Softmax(dim=1)(outputs_target)
    avg_softmax_outs_t = torch.sum(softmax_outs_t, dim=0) / float(batch_size)
    log_avg_softmax_outs_t = torch.log(avg_softmax_outs_t)
    item1 = -torch.sum(avg_softmax_outs_t * log_avg_softmax_outs_t)
    item2 = -torch.sum(softmax_outs_t * torch.log(softmax_outs_t)) / float(batch_size)
    return item1 - item2

