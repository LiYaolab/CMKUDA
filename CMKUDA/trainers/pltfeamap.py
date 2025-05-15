import torch
import numpy as np
# from lib.utils.misc import NestedTensor
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import cv2
import pylab
import time
from sklearn.decomposition import PCA
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image


# class Preprocessor(object):
#     def __init__(self):
#         self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
#         self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

#     def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
#         # Deal with the image patch
#         img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
#         img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
#         # Deal with the attention mask
#         amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
#         return NestedTensor(img_tensor_norm, amask_tensor)

class Preprocessor_wo_mask(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        return img_tensor_norm


class PreprocessorX(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return img_tensor_norm, amask_tensor


class PreprocessorX_onnx(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        """img_arr: (H,W,3), amask_arr: (H,W)"""
        # Deal with the image patch
        img_arr_4d = img_arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img_arr_4d = (img_arr_4d / 255.0 - self.mean) / self.std  # (1, 3, H, W)
        # Deal with the attention mask
        amask_arr_3d = amask_arr[np.newaxis, :, :]  # (1,H,W)
        return img_arr_4d.astype(np.float32), amask_arr_3d.astype(np.bool)

def vis_attn_maps(attn_weights, q_w, k_w, skip_len, x1, x2, x1_title, x2_title, save_path='.', idxs=None, f=0):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shape1 = [q_w, q_w]
    shape2 = [k_w, k_w]

    attn_weights_mean = []
    for attn in attn_weights:
    # attn = attn_weights
    #     attn_weights_mean.append(attn.mean(dim=1).squeeze().reshape(shape1).cpu().detach().numpy())
    #     attn = attn.mean(dim=0).unsqueeze(0)
    #     attn = F.softmax(attn, dim=0)
    #     attn = F.softmax(attn, dim=1)
        attn_weights_mean.append(attn.squeeze().reshape(shape1+shape2).cpu().detach().numpy())
        # # cls_pred = cls_pred.squeeze(0)
        # h1, w1, h2, w2 = attn_weights_mean[0].shape
        # mask = torch.zeros((h1, w1, h2, w2))
        # attn_weights_mean[0] = torch.where(attn_weights_mean[0] > 0, attn_weights_mean[0], mask)
        # r = 8
        # ind = torch.nonzero(attn_weights_mean[0] == torch.max(attn_weights_mean[0]))[0]
        # mask[ind[0]-r:ind[0]+r, ind[1]-r:ind[1]+r] = 1
        # attn_weights_mean[0] = attn_weights_mean[0] * mask
        # cls_pred = cls_pred.unsqueeze(0)
        # attn_weights_mean.append(attn[..., skip_len:(skip_len+k_w**2)].mean(dim=0).squeeze().reshape(shape1+shape2).cpu())

    # downsampling factor
    fact = q_w

    # let's select 4 reference points for visualization
    # idxs = [(32, 32), (64, 64), (32, 96), (96, 96), ]
    if idxs is None:
        idxs = [(q_w, q_w)]
    block_num=0
    idx_o = idxs[0]
    for attn_weight in attn_weights_mean:
        fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        ax = fig.add_subplot(111)
        idx = (idx_o[0] // fact, idx_o[1] // fact)
        ax.imshow(attn_weight[..., idx[0], idx[1]], cmap='cividis_r', interpolation='nearest')
        # ax.imshow(attn_weight, cmap='cividis', interpolation='nearest')
        ax.axis('off')
        # ax.set_title(f'Stage2-Block{block_num}')
        plt.savefig(save_path + '/block1_attn_weight_{}.png'.format(f))
        plt.close()

    # fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    # fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    # x2_ax = fig.add_subplot(111)
    # x2 = x2.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.int32)
    # x2_ax.imshow(x2)
    # x2_ax.axis('off')
    # plt.savefig(save_path + '/{}.png'.format(x2_title))
    # plt.close()
    #
    # # the reference points as red circles
    # fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    # fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    # x1_ax = fig.add_subplot(111)
    # x1 = x1.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.int32)
    # x1_ax.imshow(x1)
    # for (y, x) in idxs:
    #     # scale = im.height / img.shape[-2]
    #     x = ((x // fact) + 0.5) * fact
    #     y = ((y // fact) + 0.5) * fact
    #     x1_ax.add_patch(plt.Circle((x, y), fact // 2, color='r'))
    #     # x1_ax.set_title(x1_title)
    #     x1_ax.axis('off')
    # plt.show()
    # plt.savefig(save_path+'/{}.png'.format(x1_title))
    # plt.close()

    del attn_weights_mean

def vis_feature_maps(img,attn_weights,save_path='/root/DAMP/output/officehome/DAMPvisualfeanew/A2R', epochn=1,batch_idxn=1,idxs=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path_hist = save_path + '/hist'
    if not os.path.exists(save_path_hist):
        os.makedirs(save_path_hist)

    # shape1 = [w, w]

    B,C,H,W = attn_weights.shape
    # attn_weights = attn_weights.view(B, C, -1)  # 形状变为 (B, C, H * W)
    # 初始化 PCA
    pca = PCA(n_components=1)
    attn_weights_mean = []  # 用于存储每个样本的 PCA 结果
    # 创建一个空的张量用于存储 PCA 结果
    # attn_weights = torch.empty(B, H, W)
    # 对每个样本进行 PCA
    for i in range(B):
        # 对每个样本应用 PCA
        # 获取当前图像的特征图
        feature_map = attn_weights[i].permute(1, 2, 0).reshape(-1, C)  # 形状为 (H/32 * W/32, 1024)
        pca_result = pca.fit_transform(feature_map.detach().cpu().numpy())  # 形状为 (H/32 * W/32, 1)
        # 将 PCA 结果恢复为特征图
        pca_image = torch.tensor(pca_result).reshape(H,W)  # 形状为 (H/32, W/32) 
        attn_weights_mean.append(pca_image)

    fact = W
    
    if idxs is None:
        idxs = [(W, W)]
    block_num=0
    idx_o = idxs[0]
    for attn_weight in attn_weights_mean:
        batch_idx=1
        fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)

        # Display it
        ax = fig.add_subplot(111)
        # idx = (idx_o[0] // fact, idx_o[1] // fact)
        ax.imshow(attn_weight, cmap='jet', interpolation='nearest')
        # ax.imshow(attn_weight, cmap='cividis', interpolation='nearest')
        ax.axis('off')
        # ax.set_title(f'Stage2-Block{block_num}')
        plt.savefig(save_path_hist + '/framecam_{}_{}_{}'.format(epochn,batch_idxn,batch_idx))
        plt.close()
        batch_idx+=1
    del attn_weights_mean


# ###########################cam
# def vis_feature_maps(imgs,attn_weights,save_path='/root/DAMP/output/officehome/DAMPvisualcam/C2A', epochn=1,batch_idxn=1,idxs=None):
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     save_path_hist = save_path + '/cam'
#     if not os.path.exists(save_path_hist):
#         os.makedirs(save_path_hist)

#     # shape1 = [w, w]

#     B,C,H,W = attn_weights.shape
#     # attn_weights = attn_weights.view(B, C, -1)  # 形状变为 (B, C, H * W)
#     # 初始化 PCA
#     pca = PCA(n_components=1)
#     attn_weights_mean = []  # 用于存储每个样本的 PCA 结果
#     # 创建一个空的张量用于存储 PCA 结果
#     # attn_weights = torch.empty(B, H, W)
#     # 对每个样本进行 PCA
#     for i in range(B):
#         # 对每个样本应用 PCA
#         # 获取当前图像的特征图
#         img=imgs[i,:,:,:]
#         feature_map = attn_weights[i].permute(1, 2, 0).reshape(-1, C)  # 形状为 (H/32 * W/32, 1024)
#         pca_result = pca.fit_transform(feature_map.detach().cpu().numpy())  # 形状为 (H/32 * W/32, 1)
#         # 将 PCA 结果恢复为特征图
#         pca_image = torch.tensor(pca_result).reshape(H,W)  # 形状为 (H/32, W/32) 
#         batch_idx=1
#         fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
#         fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
#         ###############################
#         # Resize the CAM and overlay it
#         result = overlay_mask(to_pil_image(img), to_pil_image(pca_image), alpha=0.5)
#         ax = fig.add_subplot(111)
#         # idx = (idx_o[0] // fact, idx_o[1] // fact)
#         ax.imshow(result, cmap='jet', interpolation='nearest')
#         # ax.imshow(attn_weight, cmap='cividis', interpolation='nearest')
#         ax.axis('off')
#         ##############################################

#         # # Display it
#         # ax = fig.add_subplot(111)
#         # # idx = (idx_o[0] // fact, idx_o[1] // fact)
#         # ax.imshow(attn_weight, cmap='jet', interpolation='nearest')
#         # # ax.imshow(attn_weight, cmap='cividis', interpolation='nearest')
#         # ax.axis('off')
#         # # ax.set_title(f'Stage2-Block{block_num}')
#         plt.savefig(save_path_hist + '/framecam_{}_{}_{}'.format(epochn,batch_idxn,batch_idx))
#         plt.close()
#         batch_idx+=1
#     del attn_weights_mean
