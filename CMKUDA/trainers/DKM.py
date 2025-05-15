import torch.nn.functional as F
import torch.nn as nn
import torch
import random
import torch.nn.utils.weight_norm as weightNorm

def kl_div_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = (q *logq).sum(dim=1).mean(dim=0)
    qlogp = (q *logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp

def entropy(input):
    epsilon = 1e-5
    entropy = -input * torch.log(input + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy.mean()

def gentropy(softmax_out):
    epsilon = 1e-5
    msoftmax = softmax_out.mean(dim=0)
    gentropy = -msoftmax * torch.log(msoftmax + epsilon)
    return torch.sum(gentropy)

def ent_loss(out):
    # out: BEFORE softmax
    softmax_out = nn.Softmax(dim=1)(out)
    entropy_loss = entropy(softmax_out) - gentropy(softmax_out)
    return entropy_loss

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


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="bn"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x #torch.Size([96, 256])

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="wn"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x
    
class evidence_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(evidence_classifier, self).__init__()
        self.type = type
        self.activation = nn.Softplus()
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        x=self.activation(x)
        return x

class weight_generator(nn.Module):
    def __init__(self, fea_dim=1024):
        super(weight_generator, self).__init__()
        self.fc1 = nn.Linear(fea_dim, 256)
        #self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.activate = nn.Sigmoid()

    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def reset_random_parameters(self, reset_probability=0.5):
        for name, param in self.named_parameters():
            if random.random() < reset_probability:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x):
        x = self.fc1(x)
        #x = self.relu(x)
        x = self.fc2(x)
        x = self.activate(x)
        return x
    
class ModalitySeperation(nn.Module):
    def __init__(self, fea_dim) -> None:
        super().__init__()
        self.vis_proj = nn.Linear(fea_dim, fea_dim)
        self.txt_proj = nn.Linear(fea_dim, fea_dim)

    def forward(self, x):
        vis_fea = self.vis_proj(x)
        txt_fea = self.txt_proj(x)
        return vis_fea, txt_fea


class Discriminator(nn.Module):
    def __init__(self, fea_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(fea_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x
    


def ce_loss(label, alpha, c, global_step=1, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(label, num_classes=c)
    A = torch.sum(
        label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True
    )

    annealing_coef = global_step / annealing_step
    alp = E * (1 - label) + 1
    B = KL(alp, c)
    return torch.mean((A + 0.1 * B))


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl