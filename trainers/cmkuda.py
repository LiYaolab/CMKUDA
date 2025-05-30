import os.path as osp
import os
import datetime
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import MetricMeter, AverageMeter, load_pretrained_weights, load_checkpoint, save_checkpoint,count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from timm.models.layers import trunc_normal_
from dassl.data import DataManager
from dassl.data.transforms import build_transform

from .causal import causal,DomainDiscriminator
from .DKM import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from .pltfeamap import vis_feature_maps
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

_tokenizer = _Tokenizer()

def remove_and_return_new_list(input_list, element_to_remove):
    return [x for x in input_list if x != element_to_remove]

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, cfg.MODEL.BACKBONE.PATH)
    print(model_path)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    # print(model)

    return model

def IM_loss(outputs_target, mask_lt):
    outputs_target = outputs_target[mask_lt]
    batch_size = mask_lt.sum()
    softmax_outs_t = nn.Softmax(dim=1)(outputs_target)
    avg_softmax_outs_t = torch.sum(softmax_outs_t, dim=0) / float(batch_size) + 1e-5
    log_avg_softmax_outs_t = torch.log(avg_softmax_outs_t)
    item1 = -torch.sum(avg_softmax_outs_t * log_avg_softmax_outs_t)
    item2 = -torch.sum(softmax_outs_t * torch.log(softmax_outs_t + 1e-5)) / float(batch_size)
    return item2 - item1

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x) #torch.Size([64, 1, 256])
        x = x + self.self_attn(q, k, v) #torch.Size([64, 1, 256])
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x))) #torch.Size([64, 1, 256])
        return x


class ContextDecoder(nn.Module):
    def __init__(self,
                 cfg,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=512,
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        visual_dim = 1024 if cfg.MODEL.BACKBONE.NAME == 'RN50' else 512
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
                    TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
                ])
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        self.causal = causal(image_encoder_dim=transformer_width)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, text, visual):
        B, N, C = visual.shape #torch.Size([64, 32, 256]) torch.Size([64, 50, 1024])
        visual = self.memory_proj(visual) # B, N, C torch.Size([64, 32, 256])torch.Size([64, 50, 256])
        x = self.text_proj(text) # 2K, 77, C torch.Size([64, 1, 256]) torch.Size([64, 65, 256])#visda:torch.Size([32, 12, 256])
        ######################causal disentanglement torch.Size([64, 345, 256])
        f,b,e,b2 = self.causal(x) #visda:torch.Size([32, 12, 256])
        x = x + b2

        for layer in self.decoder:
            x = layer(x, visual) #torch.Size([64, 1, 256]) torch.Size([64, 65, 256])
        
        return self.out_proj(x),f,b,e

class PromptGeneratorWithDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.textdecoder = ContextDecoder(cfg)
        self.visualdecoder = ContextDecoder(cfg)

    def forward(self, x, y, modality):
        if modality == 'text':
            return self.textdecoder(x, y)
        else:
            return self.visualdecoder(x, y)
        
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        text_embedding_at_eos = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        text_embedding_all = torch.einsum('kld,dc->klc', x, self.text_projection)
        return text_embedding_at_eos, text_embedding_all
    
class ResNetImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.encoder = clip_model.visual
        self.attnpool = clip_model.visual.attnpool
        self.num_heads = self.attnpool.num_heads
        self.embed_dim = self.attnpool.k_proj.in_features
        self.spacial_dim = self.encoder.input_resolution // 32
        self.relu = nn.ReLU(inplace=True)
        self.out_indices=(0, 1, 2, 3)
        
    @autocast()
    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.encoder.conv1, self.encoder.bn1), (self.encoder.conv2, self.encoder.bn2), (self.encoder.conv3, self.encoder.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.encoder.avgpool(x)
            return x

        x = x.type(self.encoder.conv1.weight.dtype)
        x = stem(x)

        outs = []
        x = self.encoder.layer1(x)
        outs.append(x)
        x = self.encoder.layer2(x)
        outs.append(x)
        x = self.encoder.layer3(x)
        outs.append(x)
        x = self.encoder.layer4(x)
        outs.append(x)

        B, C, H, W = x.shape 

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        cls_pos = self.attnpool.positional_embedding[0:1, :]
        spatial_pos = F.interpolate(self.attnpool.positional_embedding[1:,].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.attnpool.q_proj.weight,
            k_proj_weight=self.attnpool.k_proj.weight,
            v_proj_weight=self.attnpool.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.attnpool.q_proj.bias, self.attnpool.k_proj.bias, self.attnpool.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.attnpool.c_proj.weight,
            out_proj_bias=self.attnpool.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        # NC(1+HW)
        x = x.permute(1, 2, 0)
        x_global = x[:, :, 0] #NC
        x_local = x[:, :, 1:].reshape(B, -1, H, W)# NCHW

        final_outs = []
        for i in self.out_indices:
            final_outs.append(outs[i])
        final_outs.append([x_global, x_local])
        return tuple(final_outs)

class VITImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.encoder = clip_model.visual
        
    @autocast()
    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)  # shape = [*, width, grid, grid]
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.encoder.positional_embedding.to(x.dtype)
        x = self.encoder.ln_pre(x)
        features.append(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        features.append(x)
        x = self.encoder.ln_post(x)
        features.append(x)
        if self.encoder.proj is not None:
            x = x @ self.encoder.proj

        global_embedding = x[:, 0]
        visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2) # B C H W

        features.append([global_embedding, visual_embedding])
        return tuple(features)

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.CMKUDA.N_CTX

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        n_dm = len(cfg.DATASET.SOURCE_DOMAINS) + len(
            cfg.DATASET.TARGET_DOMAINS)  # number of domains
        n_lencls = cfg.TRAINER.CMKUDA.N_CLS

        n = n_ctx # number of learnable tokens
        self.n_dm = n_dm
        self.n_lencls = n_lencls 
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        naive_prompt_prefix = f'a {cfg.DATASET.TARGET_DOMAINS[0]} photo of a'.replace("_", " ")
        # print(naive_prompt_prefix_len)
        # define the learnable prompt 
        if cfg.TRAINER.CMKUDA.CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        print("ctx vectors size: ".format(ctx_vectors.size()))

        self.gamma_t = nn.Parameter(torch.ones(1) * 0.01)
        self.gamma_v = nn.Parameter(torch.ones(1) * 0.01)
        prompt_prefix = " ".join(["X"] * n)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of cls words (tokens): {n_lencls}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        naive_prompts = [
            naive_prompt_prefix + " " + name + "." for name in classnames
        ]
        prompts = [
            prompt_prefix + " " + name + "." for name in classnames
        ]        
        print(f'Prompts: "{prompts[0]}"')
        print(f'Naive Prompts: "{naive_prompts[0]}"')
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        naive_tokenized_prompts = torch.cat([clip.tokenize(p) for p in naive_prompts])
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)   # cls, 77, 512  
            naive_embedding = clip_model.token_embedding(naive_tokenized_prompts).type(dtype) # cls, 77, 512

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # tokenized_prompts = torch.cat(
        #     [tokenized_prompts, naive_tokenized_prompts])
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1+n:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = cfg.TRAINER.CMKUDA.CSC
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.naive_tokenized_prompts = naive_tokenized_prompts
        self.name_lens = name_lens
        self.naive_embedding = naive_embedding

        backbone_name = cfg.MODEL.BACKBONE.NAME
        if backbone_name == 'RN50':
            bobottleneck_dim = 1024
        elif backbone_name == 'ViT-B/16' or cfg.MODEL.BACKBONE.NAME == 'RN101':
            bobottleneck_dim = 512
        elif backbone_name == 'ViT-L/14':
            bobottleneck_dim = 768
        self.fea_dim = bobottleneck_dim #1024
        self.bottleneck_dim=256
        self.clf = feat_classifier(self.n_cls, type='wn')
        self.weight = weight_generator(self.fea_dim) #两个线性层
        self.mos = ModalitySeperation(self.fea_dim)
        self.discri = Discriminator(self.fea_dim)
        self.bottleneck = feat_bottleneck(self.fea_dim, type='bn')
        self.evidence = evidence_classifier(class_num=self.n_cls,bottleneck_dim=self.bottleneck_dim,type='linear')

    @autocast()
    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # cls 16 512, broadcast to all classes
        prompts = torch.cat([
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, context_feature = 'attention', use_visual_prompt_generator=True, use_text_prompt_generator=True):
        super().__init__()
        backbone_name = cfg.MODEL.BACKBONE.NAME
        
        self.image_encoder = ResNetImageEncoder(clip_model) if backbone_name == 'RN101' or cfg.MODEL.BACKBONE.NAME == 'RN50' else VITImageEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.context_decoder = ContextDecoder(cfg) #PromptGeneratorWithDecoder(cfg) #ContextDecoder(cfg)
        # print(self.context_decoder)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.context_feature = context_feature
        self.use_visual_prompt_generator = use_visual_prompt_generator
        self.use_text_prompt_generator = use_text_prompt_generator
        self.naive_text_embedding = self.text_encoder(self.prompt_learner.naive_embedding, self.prompt_learner.naive_tokenized_prompts)[0].to(
            torch.device("cuda")) # text features for zero-shot pseudo labels
        # tunable = sum(p.numel() for p in self.image_encoder.parameters())
        # print("tunable parameters: ", tunable)


    @autocast()
    def forward(self, img, ind=False, pse=False, fea = False,featuremap=False,cam=False):
        # 4 stage output + (global feat, self-attention based feat)
        x = self.image_encoder(img) 
        # BC         BCHW
        global_feat, visual_embeddings = x[-1]
        vis_fea, txt_fea = self.prompt_learner.mos(global_feat)
        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            # (B, C, 1+H*W)
            # visual_contexts = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],
            #                            dim=2).permute(0, 2, 1)  # B, (1+H*W), C
            visual_contexts = torch.cat([txt_fea.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],
                                       dim=2).permute(0, 2, 1)  # B, (1+H*W), C
        raw_prompt = self.prompt_learner() # K, 77, 512
        tokenized_prompts = self.tokenized_prompts # K, 77, 512 
        K = self.prompt_learner.n_cls
        text_embeddings, text_contexts = self.text_encoder(raw_prompt, tokenized_prompts)
        text_embeddings = text_embeddings.expand(B, -1, -1) # B, K, C torch.Size([64, 65, 1024])
        text_contexts = text_contexts.expand(B, -1, -1, -1)[:,0,:self.prompt_learner.n_ctx,:] # B, L, C torch.Size([64, 32, 1024])
        # update visual prompting
        if self.use_visual_prompt_generator:
            # update visual_embeddings by text_context, post-model prompting refines the visual_embeddings
            # visual_embeddings: # (B, 1, C) text_contexts: B, (L-1), C
            vis_prompt_diff,f,b,e = self.context_decoder(global_feat.reshape(B, C, 1).permute(0, 2, 1), text_contexts)
            vis_prompt_diff = vis_prompt_diff.permute(0, 2, 1).reshape(B, C)
            updated_vision_embedding = global_feat + self.prompt_learner.gamma_v * vis_prompt_diff
        
        # update text prompting
        if self.use_text_prompt_generator:
            # update text_embeddings by visual_context, post-model prompting refines the text_embeddings
            # text_embeddings: # (B, K, C) visual_contexts: B, (1+H*W), C,torch.Size([64, 65, 1024])torch.Size([64, 50, 1024])
            text_diff ,f2,b2,e2 = self.context_decoder(text_embeddings, visual_contexts)
            # (B, K, C) 
            updated_text_embeddings = text_embeddings + self.prompt_learner.gamma_t * text_diff
        
        f = f+f2
        b = b+b2
        e = e+e2

        return_all = [] 
        # compute logits based on updated embeddings (modality matching)
        visual = F.normalize(updated_vision_embedding, dim=1, p=2)
        text = F.normalize(updated_text_embeddings, dim=2, p=2)
        logits = torch.einsum('bc,bkc->bk', visual, text)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * logits # logits
        return_all.append(logits)

        if ind:
            # compute logits based on updated embeddings (individuation)
            # duplicate_visual = visual.expand(B, B, C)
            logits_individuation = torch.einsum('ac,bkc->abk', visual, text).mean(dim=-1) ## B, B
            logits_individuation = logits_individuation * logit_scale #/ self.prompt_learner.contrastive_T
            return_all.append(logits_individuation)
        if pse:
            # compute logits based on original embeddings for pseudo labeling
            image_features = global_feat
            text_features = self.naive_text_embedding
            image_features = image_features / image_features.norm(dim=-1,
                                                                keepdim=True) # B, C
            text_features = text_features / text_features.norm(dim=-1,
                                                            keepdim=True) # K, C        
            pseudo_logits = logit_scale * image_features @ text_features.t()
            return_all.append(pseudo_logits)
        if fea:
            return_all.append(global_feat)
            return_all.append(updated_vision_embedding)
            return_all.append(updated_text_embeddings)
        
        return_all.append(vis_fea)
        return_all.append(txt_fea)
        return_all.append(f)
        return_all.append(b)
        return_all.append(e)
        if featuremap:
            # return_all.append(x)
            return_all.append(visual_embeddings)
            return_all.append(text_embeddings)
        # if cam:
        #     return_all.append(x[3])

        return tuple(return_all)

@TRAINER_REGISTRY.register()
class CMKUDA(TrainerXU):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.CMKUDA.PREC in ["fp16", "fp32", "amp"]

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.CMKUDA.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        self.dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes
        self.lab2cname = self.dm.lab2cname

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        input_x2 = batch_x["img2"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]
        input_u2 = batch_u["img2"]
        # label_u is used only for evaluating pseudo labels' accuracy
        label_u = batch_u["label"]

        input_x = input_x.to(self.device)
        input_x2 = input_x2.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)
        input_u2 = input_u2.to(self.device)
        label_u = label_u.to(self.device)

        return input_x, input_x2, label_x, input_u, input_u2, label_u

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(classnames)
        self.featuremap = False

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.CMKUDA.PREC == "fp32" or cfg.TRAINER.CMKUDA.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        # plus one for pseudo label
        self.n_cls = self.model.prompt_learner.n_cls
        # self.smooth_CE = CrossEntropyLabelSmooth(self.n_cls)

        #############################causal DomainDiscriminator
        hidden_dim = 128
        backbone_name = cfg.MODEL.BACKBONE.NAME
        if cfg.MODEL.BACKBONE.NAME == 'RN50':
            bobottleneck_dim = 256
        elif backbone_name == 'ViT-B/16' or backbone_name == 'RN101':
            bobottleneck_dim = 256
        elif backbone_name == 'ViT-L/14':
            bobottleneck_dim = 256
        # bottleneck_dim = 512 #256
        self.domain_discriminator = DomainDiscriminator(bobottleneck_dim, hidden_dim)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "context_decoder" not in name:
                param.requires_grad_(False)
        
        ########################
        # Double check
        # enabled = set()
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         enabled.add(name)
        # print(f"Parameters to be updated: {sorted(enabled)}")
        # 计算 prompt_learner 的参数量
        prompt_learner_params = count_num_param(self.model.prompt_learner)
        # 计算 context_decoder 的参数量
        context_decoder_params = count_num_param(self.model.context_decoder)
        # 计算两部分总共的参数量
        total_params = prompt_learner_params + context_decoder_params
        print("# params: {:,}".format(total_params))
 
        if cfg.MODEL.INIT_WEIGHTS_PRO:
            load_pretrained_weights(self.model.prompt_learner,
                                    cfg.MODEL.INIT_WEIGHTS_PRO)
        if cfg.MODEL.INIT_WEIGHTS_CTX:
            load_pretrained_weights(self.model.context_decoder,
                                    cfg.MODEL.INIT_WEIGHTS_CTX)

        self.model.to(self.device)
        
        self.domain_discriminator.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim_p = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched_p = build_lr_scheduler(self.optim_p, cfg.OPTIM)

        self.optim_c = build_optimizer(self.model.context_decoder, cfg.OPTIM_C)
        self.sched_c = build_lr_scheduler(self.optim_c, cfg.OPTIM_C)

        self.optim_dom = build_optimizer(self.domain_discriminator, cfg.OPTIM)
        self.sched_dom = build_lr_scheduler(self.optim_dom, cfg.OPTIM)

        '''
        register model could be updated. When new module needs to be updated
        register the module before use
        '''
        self.register_model("prompt_learner", self.model.prompt_learner,
                            self.optim_p, self.sched_p)
        
        self.register_model("context_decoder", self.model.context_decoder,
                                    self.optim_c, self.sched_c)

        self.register_model("domain_discriminator", self.domain_discriminator,
                                    self.optim_dom, self.sched_dom)

        self.scaler = GradScaler() if cfg.TRAINER.CMKUDA.PREC == "amp" else None

        

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def train(self):
        """Generic training loops."""

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()
        


    def run_epoch(self):
        self.threshold = self.cfg.TRAINER.CMKUDA.TAU #+ (0.8-self.cfg.TRAINER.CMKUDA.TAU) * self.epoch / (self.max_epoch - self.epoch) 
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x if self.cfg.DATASET.NAME == "OfficeHome" else 500
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        # self.test_batches = [int(self.num_batches * 0.33), int(self.num_batches * 0.66)]
        # self.data_src_go=[]
        # self.data_src_vi=[]
        # self.data_src_te=[]
        # self.data_tar_go=[]
        # self.data_tar_vi=[]
        # self.data_tar_te=[]
        # self.label_src=[]
        # self.label_tar=[]
        # self.label_tar_pseudo=[]
        # self.data_src_yuanfeav=[]
        # self.data_tar_yuanfeav=[]
        # self.data_src_yuanfeat=[]
        # self.data_tar_yuanfeat=[]

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (
                    self.batch_idx + 1
            ) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch -
                              1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print("epoch [{0}/{1}][{2}/{3}]\t"
                      "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      "eta {eta}\t"
                      "{losses}\t"
                      "lr {lr:.6e}".format(
                          self.epoch + 1,
                          self.max_epoch,
                          self.batch_idx + 1,
                          self.num_batches,
                          batch_time=batch_time,
                          data_time=data_time,
                          eta=eta,
                          losses=losses,
                          lr=self.get_current_lr(),
                      ))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
        # data_src_yuanfeav_npy=torch.cat(self.data_src_yuanfeav,axis=0).detach().cpu().numpy()
        # data_tar_yuanfeav_npy=torch.cat(self.data_tar_yuanfeav,axis=0).detach().cpu().numpy()
        # data_src_yuanfeat_npy=torch.cat(self.data_src_yuanfeav,axis=0).detach().cpu().numpy()
        # data_tar_yuanfeat_npy=torch.cat(self.data_tar_yuanfeat,axis=0).detach().cpu().numpy()
        
        # data_src_go_npy=torch.cat(self.data_src_go,axis=0).detach().cpu().numpy()
        # data_src_vi_npy=torch.cat(self.data_src_vi,axis=0).detach().cpu().numpy()
        # data_src_te_npy=torch.cat(self.data_src_te,axis=0).detach().cpu().numpy()
        # data_tar_go_npy=torch.cat(self.data_tar_go,axis=0).detach().cpu().numpy()
        # data_tar_vi_npy=torch.cat(self.data_tar_vi,axis=0).detach().cpu().numpy()
        # data_tar_te_npy=torch.cat(self.data_tar_te,axis=0).detach().cpu().numpy()
        # label_src_npu=torch.cat(self.label_src,axis=0).detach().cpu().numpy()
        # label_tar_npu=torch.cat(self.label_tar,axis=0).detach().cpu().numpy()
        # label_tarpse_npu=torch.cat(self.label_tar_pseudo,axis=0).detach().cpu().numpy()
        # save_path_hist='/root/CMKUDA/output/officehome/CMKUDAvisualtsne/A2Ryuanfea/tsne'
        # if not os.path.exists(save_path_hist):
        #     os.makedirs(save_path_hist)
        # np.save(save_path_hist + '/data_src_yuanfeav_npy_{}.npy'.format(self.epoch+1),data_src_yuanfeav_npy)
        # np.save(save_path_hist + '/data_tar_yuanfeav_npy_{}.npy'.format(self.epoch+1),data_tar_yuanfeav_npy)
        # np.save(save_path_hist + '/data_src_yuanfeat_npy_{}.npy'.format(self.epoch+1),data_src_yuanfeat_npy)
        # np.save(save_path_hist + '/data_tar_yuanfeat_npy_{}.npy'.format(self.epoch+1),data_tar_yuanfeat_npy)
        # np.save(save_path_hist + '/data_src_go_npy_{}.npy'.format(self.epoch+1),data_src_go_npy)
        # np.save(save_path_hist + '/data_src_vi_npy_{}.npy'.format(self.epoch+1),data_src_vi_npy)
        # np.save(save_path_hist + '/data_src_te_npy_{}.npy'.format(self.epoch+1),data_src_te_npy)
        # np.save(save_path_hist + '/data_tar_go_npy_{}.npy'.format(self.epoch+1),data_tar_go_npy)
        # np.save(save_path_hist + '/data_tar_vi_npy_{}.npy'.format(self.epoch+1),data_tar_vi_npy)
        # np.save(save_path_hist + '/data_tar_te_npy_{}.npy'.format(self.epoch+1),data_tar_te_npy)
        # np.save(save_path_hist + '/label_src_npu_{}.npy'.format(self.epoch+1),label_src_npu)
        # np.save(save_path_hist + '/label_tar_npu_{}.npy'.format(self.epoch+1),label_tar_npu)
        # np.save(save_path_hist + '/label_tarpse_npu_{}.npy'.format(self.epoch+1),label_tarpse_npu)

    def forward_backward(self, batch_x, batch_u):
        # label_u only used for matric
        image_x, image_x2, label, image_u, image_u2, label_u = self.parse_batch_train(batch_x, batch_u)
        prec = self.cfg.TRAINER.CMKUDA.PREC
        if prec == "amp":
            with autocast():
                output_x, output_x_ind,src_vis_fea,src_txt_fea, f_s,b_s,e_s = self.model(image_x, ind =True, pse=False)
                output_u, output_u_ind, pseudo_label_logits,tar_vis_fea,tar_txt_fea, f_t,b_t,e_t = self.model(image_u, ind=True, pse=True)
                output_x2 = self.model(image_x2)[0]
                output_u2 = self.model(image_u2)[0]
                ###################
                # self.data_src_yuanfeav.append(src_vis_fea)
                # self.data_tar_yuanfeav.append(tar_vis_fea)
                # self.data_src_yuanfeat.append(v1)
                # self.data_tar_yuanfeat.append(v2)
                # self.data_src_go.append(global_feat)
                # self.data_src_vi.append(vi)
                # self.data_src_te.append(te)
                # self.data_tar_go.append(global_feat2)
                # self.data_tar_vi.append(vi2)
                # self.data_tar_te.append(te2)
                # self.label_src.append(label)
                # self.label_tar.append(label_u)
                # self.label_tar_pseudo.append(pseudo_label_logits)
                ######################################causal disentanglement
                # feature combine
                fs_ori = b_s + e_s
                ft_ori = b_t + e_t
                fs_rec = b_s + e_t
                ft_rec = b_t + e_s
                
                # domain classification loss
                f_ori = torch.cat([fs_ori, ft_ori], dim=0)
                f_rec = torch.cat([ft_rec, fs_rec], dim=0)
                batch_size = output_x.size(0)     
                domain_label = torch.from_numpy(np.array([[0.]] * batch_size + [[1.0]] * batch_size)).float().cuda()
                domain_label_ori_hat = self.domain_discriminator(f_ori)
                domain_label_rec_hat = self.domain_discriminator(f_rec)
                domain_loss =  nn.BCEWithLogitsLoss()(domain_label_ori_hat, domain_label) \
                            +  nn.BCEWithLogitsLoss()(domain_label_rec_hat, domain_label)
                domain_loss = 2.0 * domain_loss
                # dis_loss = 5.0 *  torch.mean(torch.abs(F.softmax(self.model.prompt_learner.causal.head1(f_t)) - F.softmax(self.model.prompt_learner.causal.head2(ft_rec))))
                # cls_disparency_loss = 5.0 * calc_similiar_penalty(self.model.prompt_learner.causal.head1, self.model.prompt_learner.causal.head2)
                causaldisentangle_loss = domain_loss #+ dis_loss + cls_disparency_loss

                #######UniMos
                src_len = output_x.shape[0]
                tar_len = output_u.shape[0]
                vis_fea = torch.concat([src_vis_fea,tar_vis_fea])
                bottleneck_fea = self.model.prompt_learner.bottleneck(vis_fea)
                vis_out = self.model.prompt_learner.clf(bottleneck_fea) 
                src_vis_out, tar_vis_out = vis_out[:src_len], vis_out[src_len:] 
                src_txt_out = output_x
                tar_txt_out = output_u
                clf_l = self.model.prompt_learner.weight(tar_vis_fea).mean()
                clf_p = 1 - clf_l
                ens_tar_vis_out = (clf_l*tar_vis_out + clf_p*(tar_txt_out)) #tar_txt_out.detach()
                # ens_src_vis_out = (clf_l*src_vis_out + clf_p*(src_txt_out))
                ############################
                
                # only clip annotation
                mix_lambda = self.epoch / self.max_epoch
                pseudo_label = (torch.softmax(output_u.reshape(-1, self.n_cls), dim=-1) * mix_lambda + torch.softmax(pseudo_label_logits.reshape(-1, self.n_cls), dim=-1) * (1-mix_lambda)).detach()  
            
                max_probs, label_p = torch.max(pseudo_label, dim=-1)
                mask_ge = max_probs.ge(self.threshold).float()
                mask_ge_bool = max_probs.ge(self.threshold)
                mask_lt = max_probs.lt(1.0)

                ################evidence
                evidence = self.model.prompt_learner.evidence(bottleneck_fea)
                alpha = evidence + 1
                loss_evi = 1.0*(ce_loss(label, alpha[:src_len], self.n_cls) + ce_loss(label_p, alpha[src_len:], self.n_cls))
                ##############################

                #######UniMosloss
                vis_celoss = 1.0 * F.cross_entropy(src_vis_out, label) + F.cross_entropy(ens_tar_vis_out, label_p)
                #txt_celoss = 1.0 * F.cross_entropy(src_txt_out, label) + kl_div_with_logit(tar_txt_out, pseudo_label)
                sep_loss = 0.01 * ((src_txt_fea @ src_vis_fea.T).diag().pow(2).sum() + (tar_txt_fea @ tar_vis_fea.T).diag().pow(2).sum())
                # imloss = ent_loss(ens_tar_vis_out) #公式12
                imloss = IM_loss(ens_tar_vis_out, mask_lt)

                # loss_UniMos = sum(loss_dict.values())
                loss_UniMos = vis_celoss + sep_loss + imloss #+ txt_celoss 
                # output_u = ens_tar_vis_out 

                loss_x = F.cross_entropy(output_x, label)
                loss_x2 = F.cross_entropy(output_x2, label)
                loss_u = torch.tensor(0.0).cuda() if mask_ge.sum() == 0 else (F.cross_entropy(output_u, label_p, reduction='none') * mask_ge).sum() / mask_ge.sum()
                loss_u2 = torch.tensor(0.0).cuda() if mask_ge.sum() == 0 else (F.cross_entropy(output_u2, label_p, reduction='none') * mask_ge).sum() / mask_ge.sum()

                x_ind_label = torch.arange(self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE, dtype=torch.long).to(self.device)
                loss_x_ind = (F.cross_entropy(output_x_ind, x_ind_label) + F.cross_entropy(output_x_ind.permute(1, 0), x_ind_label)) / 2.0 

                u_ind_label = torch.arange(self.cfg.DATALOADER.TRAIN_U.BATCH_SIZE, dtype=torch.long).to(self.device)
                loss_u_ind = (F.cross_entropy(output_u_ind, u_ind_label) + F.cross_entropy(output_u_ind.permute(1, 0), u_ind_label)) / 2.0

                loss_ind = (loss_x_ind + loss_u_ind)
                
                loss_im = IM_loss(output_u, mask_lt)

                loss = (loss_x + loss_x2) + self.cfg.TRAINER.CMKUDA.U * (loss_u + loss_u2) + loss_ind + loss_im + causaldisentangle_loss + loss_UniMos +loss_evi

                self.optim_p.zero_grad()
                self.optim_c.zero_grad()
                self.optim_dom.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim_p)
                self.scaler.step(self.optim_c)
                self.scaler.step(self.optim_dom)
                # if(self.epoch %2 == 0):
                #     self.scaler.step(self.optim_dom)
                self.scaler.update()

        loss_summary = {
            "loss":
            loss.item(),
            "loss_x":
            loss_x.item(),
            "loss_u":
            loss_u.item(),
            "acc_x":
            compute_accuracy(output_x, label)[0].item(),
            "acc_u":
            compute_accuracy(pseudo_label, label_u)[0].item(),
            "gamma_v":
            self.model.prompt_learner.gamma_v,
            "gamma_t":
            self.model.prompt_learner.gamma_t,
            # "contrastive_T":
            # self.model.prompt_learner.contrastive_T
            "loss_x_ind":
            loss_x_ind,
            "causaldisentangle_loss":
            causaldisentangle_loss,
            "loss_UniMos":
            loss_UniMos,
            "loss_evi":
            loss_evi,
            # "loss_u_ind":
            # loss_u_ind,
            # "loss_im":
            # loss_im
        }

        self.update_lr()

        return loss_summary

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) %
                                self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if
                                self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test:
            curr_result = self.test()
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(self.epoch,
                                self.output_dir,
                                model_name="model-best.pth.tar")

            self.set_model_mode("train")

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained model is given"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} "
                  'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        if split is None:
            split = self.cfg.TEST.SPLIT
        # # ##############################
        # self.all_true_labels = []  # 用于存储所有真实标签
        # self.all_predicted_labels = []  # 用于存储所有预测标签

        data_loader = self.test_loader
        print("Do evaluation on test set")
        featuremap = False
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output,_,_,_,_,_ = self.model_inference(input,featuremap=featuremap)
            # vis_feature_maps(input,attn_weights=x[0],epochn=self.epoch,batch_idxn=batch_idx)
            self.evaluator.process(output, label)
            # ###########################confusem
            # # 处理输出和标签
            # predicted_labels = np.argmax(output.cpu().numpy(), axis=1)  # 移动到 CPU 并转换为 NumPy 数组
            # self.all_true_labels.extend(label.cpu().numpy())  # 存储真实标签
            # self.all_predicted_labels.extend(predicted_labels)  # 存储预测标签
        featuremap = False
        # # 计算混淆矩阵
        # cm = confusion_matrix(self.all_true_labels, self.all_predicted_labels)
        # save_path_hist='/root/CMKUDA/output/visda2017/CMKUDAvisualconfusem/A2R/confuse'
        # if not os.path.exists(save_path_hist):
        #     os.makedirs(save_path_hist)
        # # 可视化混淆矩阵
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # disp.plot(cmap=plt.cm.Blues)  # 使用蓝色调的配色
        # # 隐藏坐标轴标签
        # plt.xlabel('')  # 隐藏x轴标签
        # plt.ylabel('')  # 隐藏y轴标签
        # plt.figure(figsize=(8, 6))
        # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # 使用插值使背景颜色
        # plt.title('Confusion Matrix')
        # plt.colorbar()  # 添加颜色条
        # plt.savefig(save_path_hist+'/frameconfuse_{}.png'.format(self.epoch))

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        results_all = results["accuracy"]

        return results_all