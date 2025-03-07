from typing import Sequence, Tuple, Type, Union
#from .bridger import Bridger_RN as Bridger_RL, Bridger_ViT as Bridger_VL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from model.SwinUNETR import SwinUNETR
from model.Unet import UNet3D
from model.unetr_pp import UNETR_PP
#from model.DiNTS import TopologyInstance, DiNTS
#from model.Unetpp import BasicUNetPlusPlus
#from model.clip import build_model
import scipy.fft as fft
from torch.cuda.amp import autocast
from typing import List, Union
import math 
#from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.utils import ORGAN_NAME
from timm.models.layers import trunc_normal_



def add_bias_towards_void(query_class_logits, void_prior_prob=0.9):
    class_logits_shape = query_class_logits.shape
    init_bias = [0.0] * class_logits_shape[-1]
    init_bias[-1] = math.log(
      (class_logits_shape[-1] - 1) * void_prior_prob / (1 - void_prior_prob))
    return query_class_logits + torch.tensor(init_bias, dtype=query_class_logits.dtype).to(query_class_logits)



def get_activation(name):
    if name is None or name.lower() == 'none':
        return nn.Identity()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()

def get_norm3d(name, channels):
    if name is None or name.lower() == 'none':
        return nn.Identity()

    if name.lower() == 'syncbn':
        return nn.BatchNorm3d(channels, eps=1e-3, momentum=0.01)

def get_norm2d(name, channels):
    if name is None or name.lower() == 'none':
        return nn.Identity()

    if name.lower() == 'syncbn':
        return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)

def get_norm1d(name, channels):
    if name is None or name.lower() == 'none':
        return nn.Identity()

    if name.lower() == 'syncbn':
        return nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)

class AttentionOperation(nn.Module):
    def __init__(self, channels_v, num_heads):
        super().__init__()
        self._batch_norm_similarity = get_norm2d('syncbn', num_heads)
        self._batch_norm_retrieved_value = get_norm1d('syncbn', channels_v)

    def forward(self, query, key, value):
        N, _, _, L = query.shape
        _, num_heads, C, _ = value.shape
        similarity_logits = torch.einsum('bhdl,bhdm->bhlm', query, key)
        similarity_logits = self._batch_norm_similarity(similarity_logits)

        with autocast(enabled=False):
            attention_weights = F.softmax(similarity_logits.float(), dim=-1)
        retrieved_value = torch.einsum(
            'bhlm,bhdm->bhdl', attention_weights, value)
        retrieved_value = retrieved_value.reshape(N, num_heads * C, L)
        retrieved_value = self._batch_norm_retrieved_value(
            retrieved_value)
        retrieved_value = F.gelu(retrieved_value)
        return retrieved_value


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm=None, act=None,
                 conv_type='3d', conv_init='he_normal', norm_init=1.0):
        super().__init__()

        if conv_type == '2d':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif conv_type == '1d':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif  conv_type == '3d':
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.norm = get_norm3d(norm, out_channels)
        self.act = get_activation(act)

        if conv_init == 'normal':
            nn.init.normal_(self.conv.weight, std=.02)
        elif conv_init == 'trunc_normal':
            trunc_normal_(self.conv.weight, std=.02)
        elif conv_init == 'he_normal':
            # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
            trunc_normal_(self.conv.weight, std=math.sqrt(2.0 / in_channels))
        elif conv_init == 'xavier_uniform':
            nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)

        if norm is not None:
            nn.init.constant_(self.norm.weight, norm_init)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvBN1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm=None, act=None,
                 conv_type='3d', conv_init='he_normal', norm_init=1.0):
        super().__init__()

        if conv_type == '2d':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif conv_type == '1d':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif  conv_type == '3d':
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.norm = get_norm1d(norm, out_channels)
        self.act = get_activation(act)

        if conv_init == 'normal':
            nn.init.normal_(self.conv.weight, std=.02)
        elif conv_init == 'trunc_normal':
            trunc_normal_(self.conv.weight, std=.02)
        elif conv_init == 'he_normal':
            # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
            trunc_normal_(self.conv.weight, std=math.sqrt(2.0 / in_channels))
        elif conv_init == 'xavier_uniform':
            nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)

        if norm is not None:
            nn.init.constant_(self.norm.weight, norm_init)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class kMaXTransformerLayer(nn.Module):
    def __init__(
        self,
        num_classes=133,
        in_channel_pixel=2048,
        in_channel_query=256,
        base_filters=128,
        num_heads=8,
        bottleneck_expansion=2,
        key_expansion=1,
        value_expansion=2,
        drop_path_prob=0.0,
        t_max=0.4
    ):
        super().__init__()
        self.t_max = t_max
        self._num_classes = num_classes
        print('self._num_classes:{}'.format(self._num_classes))
        self._num_heads = num_heads
        self._bottleneck_channels = int(round(base_filters * bottleneck_expansion))
        self._total_key_depth = int(round(base_filters * key_expansion))
        self._total_value_depth = int(round(base_filters * value_expansion))

        # Per tf2 implementation, the same drop path prob are applied to:
        # 1. k-means update for object query
        # 2. self/cross-attetion for object query
        # 3. ffn for object query
        self.drop_path_kmeans = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity() 
        self.drop_path_attn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity() 
        self.drop_path_ffn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity() 

        initialization_std = self._bottleneck_channels ** -0.5
        self._query_conv1_bn_act = ConvBN1d(in_channel_query, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='syncbn', act='gelu', conv_type='1d')

        self._pixel_conv1_bn_act = ConvBN(in_channel_pixel, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='syncbn', act='gelu')

        self._query_qkv_conv_bn = ConvBN1d(self._bottleneck_channels, self._total_key_depth * 2 + self._total_value_depth, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d')
        trunc_normal_(self._query_qkv_conv_bn.conv.weight, std=initialization_std)

        self._pixel_v_conv_bn = ConvBN(self._bottleneck_channels, self._total_value_depth, kernel_size=1, bias=False,
                                          norm='syncbn', act=None)
        trunc_normal_(self._pixel_v_conv_bn.conv.weight, std=initialization_std)

        self._query_self_attention = AttentionOperation(channels_v=self._total_value_depth, num_heads=num_heads)

        self._query_conv3_bn = ConvBN1d(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d', norm_init=0.0)

        self._query_ffn_conv1_bn_act = ConvBN1d(in_channel_query, 2048, kernel_size=1, bias=False,
                                          norm='syncbn', act='gelu', conv_type='1d')
        self._query_ffn_conv2_bn = ConvBN1d(2048, in_channel_query, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d', norm_init=0.0)

        self._predcitor = kMaXPredictor(in_channel_pixel=self._bottleneck_channels,
            in_channel_query=self._bottleneck_channels, num_classes=num_classes)
        self._kmeans_query_batch_norm_retrieved_value = get_norm1d('syncbn', self._total_value_depth)
        self._kmeans_query_conv3_bn = ConvBN1d(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d', norm_init=0.0)
        
    def cal_batch_iou(self, pred,gt):
        #print('pred:{},gt:{}'.format(pred.shape, gt.shape))
        ## b, (zhw)
        batch_ratio = torch.sum(pred * gt, dim = -1).float()/ torch.sum(pred, dim = -1) 
        return batch_ratio

    def update_mask(self,index,mask, max_ratio, cluster_ind,epoch = None, thresh_min = 0.1, thresh_max = 0.4):
        # print('thresh_max:{}'.format(thresh_max))
        if epoch is not None:
            thresh = min(epoch/50.0, 1.0) * thresh_max + thresh_min
        else:
            thresh = thresh_max

       # print('epoch:{},thresh:{},max_ratio:{}'.format(epoch, thresh,max_ratio))
        for i in range(max_ratio.shape[0]):
            if max_ratio[i] < thresh:
                mask[i,index[i] == cluster_ind ] = 0
        return mask

    def forward(self, pixel_feature, query_feature, mask = None, epoch = None):
        N, C,Z, H, W = pixel_feature.shape
        if mask is not None:
            if mask.shape[-3:] != pixel_feature.shape[-3:]:
                mask = F.interpolate(mask, size= pixel_feature.shape[-3:], mode='trilinear', align_corners=True)
       #     print('mask:{}'.format(mask.shape))
            mask =mask.flatten(2).detach()
            mask_idx = mask.max(1, keepdim=True)[1]
        #print('pixel_feature.shape:{}'.format(pixel_feature.shape))
        _, D, L = query_feature.shape
        pixel_space = self._pixel_conv1_bn_act(F.gelu(pixel_feature)) # N C Z H W
        query_space = self._query_conv1_bn_act(query_feature) # N x C x L

        # k-means cross-attention.
        pixel_value = self._pixel_v_conv_bn(pixel_space) # N C Z H W
        pixel_value = pixel_value.reshape(N, self._total_value_depth, Z*H*W)

        prediction_result = self._predcitor(
            mask_embeddings=query_space, class_embeddings=query_space, pixel_feature=pixel_space)
        
        with torch.no_grad():
            clustering_result_inp = prediction_result['mask_logits'].flatten(2).detach() # N L HW
            index = clustering_result_inp.max(1, keepdim=True)[1]
            if mask is not None:
                filter_mask =torch.ones(N,1,Z*H*W).type_as(clustering_result_inp)
                for i in range(clustering_result_inp.shape[1]):
                    tmp_binary_mask = torch.zeros_like(index)
                    tmp_binary_mask[index == i] = 1
                    iou_ratios = []
                    for j in range(1,self._num_classes):
                        tmp_gt_mask = torch.zeros_like(index)
                        tmp_gt_mask[mask_idx == j] = 1
       #                 print('tmp_gt_mask:{},tmp_binary_mask:{},pixel_feature:{}'.format(tmp_gt_mask.shape, tmp_binary_mask.shape, pixel_feature.shape))
                        tmp_ratio = self.cal_batch_iou(tmp_binary_mask.squeeze(1), tmp_gt_mask.squeeze(1))
                        iou_ratios.append(tmp_ratio)
                    iou_ratios = torch.stack(iou_ratios, dim = -1)
            #        print('-------iou_ratios:{}-----'.format(iou_ratios.shape))
                    max_ratio = torch.max(iou_ratios, dim = -1)[0]
                    filter_mask = self.update_mask(index,filter_mask, max_ratio, i, epoch, thresh_max=self.t_max)
                 #   filter_mask.append(tmp_mask)
                clustering_result = torch.zeros_like(clustering_result_inp, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
                clustering_result = clustering_result * filter_mask
            else:
                clustering_result = torch.zeros_like(clustering_result_inp, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)

        with autocast(enabled=False):
        # k-means update.
            kmeans_update = torch.einsum('blm,bdm->bdl', clustering_result.float(), pixel_value.float()) # N x C x L

        kmeans_update = self._kmeans_query_batch_norm_retrieved_value(kmeans_update)
        kmeans_update = self._kmeans_query_conv3_bn(kmeans_update)
        query_feature = query_feature + self.drop_path_kmeans(kmeans_update)

        # query self-attention.
        query_qkv = self._query_qkv_conv_bn(query_space)
        query_q, query_k, query_v = torch.split(query_qkv,
         [self._total_key_depth, self._total_key_depth, self._total_value_depth], dim=1)
        query_q = query_q.reshape(N, self._num_heads, self._total_key_depth//self._num_heads, L)
        query_k = query_k.reshape(N, self._num_heads, self._total_key_depth//self._num_heads, L)
        query_v = query_v.reshape(N, self._num_heads, self._total_value_depth//self._num_heads, L)
        self_attn_update = self._query_self_attention(query_q, query_k, query_v)
        self_attn_update = self._query_conv3_bn(self_attn_update)
        query_feature = query_feature + self.drop_path_attn(self_attn_update)
        query_feature = F.gelu(query_feature)

        # FFN.
        ffn_update = self._query_ffn_conv1_bn_act(query_feature)
        ffn_update = self._query_ffn_conv2_bn(ffn_update)
        query_feature = query_feature + self.drop_path_ffn(ffn_update)
        query_feature = F.gelu(query_feature)

        return query_feature, prediction_result

class kMaXPredictor(nn.Module):
    def __init__(self, in_channel_pixel, in_channel_query, num_classes=133+1):
        super().__init__()
        self._pixel_space_head_conv0bnact = ConvBN(in_channel_pixel, in_channel_pixel, kernel_size=5, groups=in_channel_pixel, padding=2, bias=False,
                                                   norm='syncbn', act='gelu', conv_init='xavier_uniform')
        self._pixel_space_head_conv1bnact = ConvBN(in_channel_pixel, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu')
        self._pixel_space_head_last_convbn = ConvBN(256, 128, kernel_size=1, bias=True, norm='syncbn', act=None)
        trunc_normal_(self._pixel_space_head_last_convbn.conv.weight, std=0.01)

        self._transformer_mask_head = ConvBN1d(256, 128, kernel_size=1, bias=False, norm='syncbn', act=None, conv_type='1d')
        self._transformer_class_head = ConvBN1d(256, num_classes, kernel_size=1, norm=None, act=None, conv_type='1d')
        trunc_normal_(self._transformer_class_head.conv.weight, std=0.01)

        self._pixel_space_mask_batch_norm = get_norm2d('syncbn', channels=1)
        nn.init.constant_(self._pixel_space_mask_batch_norm.weight, 0.1)

    def forward(self, mask_embeddings, class_embeddings, pixel_feature):
        _,_,N = mask_embeddings.shpae
        B,_,Z,H,W = pixel_feature.shape
        pixel_space_feature = self._pixel_space_head_conv0bnact(pixel_feature)
        pixel_space_feature = self._pixel_space_head_conv1bnact(pixel_space_feature)
        pixel_space_feature = self._pixel_space_head_last_convbn(pixel_space_feature)
        pixel_space_normalized_feature = F.normalize(pixel_space_feature, p=2, dim=1)

        cluster_class_logits = self._transformer_class_head(class_embeddings).permute(0, 2, 1).contiguous()
        cluster_class_logits = add_bias_towards_void(cluster_class_logits)
        cluster_mask_kernel = self._transformer_mask_head(mask_embeddings)
        #print('pixel_space_normalized_feature:{},cluster_mask_kernel:{}'.format(pixel_space_normalized_feature.shape, cluster_mask_kernel.shape))
        mask_logits = torch.einsum('bczhw,bcn->bnzhw',
          pixel_space_normalized_feature, cluster_mask_kernel)
        mask_logits = mask_logits.view(B,N,Z*H*W)
        #b,c,z,h,w = mask_logits.shape
        
        mask_logits = self._pixel_space_mask_batch_norm(mask_logits.unsqueeze(dim=1)).squeeze(dim=1)
        mask_logits = mask_logits.view(B,N,Z,H,W)

        class_wise_mask_logits = torch.einsum('bnzhw,bnk->bkzhw',
          mask_logits, cluster_class_logits)


        return {
            'class_logits': cluster_class_logits,
            'mask_logits': mask_logits,
            'pixel_feature': pixel_space_normalized_feature,
            'class_wise_mask_logits':class_wise_mask_logits}
        


class kMaXPredictor(nn.Module):
    def __init__(self, in_channel_pixel, in_channel_query, num_classes=133+1):
        super().__init__()
        self._pixel_space_head_conv0bnact = ConvBN(in_channel_pixel, in_channel_pixel, kernel_size=5, groups=in_channel_pixel, padding=2, bias=False,
                                                   norm='syncbn', act='gelu', conv_init='xavier_uniform')
        self._pixel_space_head_conv1bnact = ConvBN(in_channel_pixel, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu')
        self._pixel_space_head_last_convbn = ConvBN(256, 128, kernel_size=1, bias=True, norm='syncbn', act=None)
        trunc_normal_(self._pixel_space_head_last_convbn.conv.weight, std=0.01)

        self._transformer_mask_head = ConvBN1d(in_channel_query, 128, kernel_size=1, bias=False, norm='syncbn', act=None, conv_type='1d')
        self._transformer_class_head = ConvBN1d(in_channel_query, num_classes, kernel_size=1, norm=None, act=None, conv_type='1d')
        trunc_normal_(self._transformer_class_head.conv.weight, std=0.01)

        self._pixel_space_mask_batch_norm = get_norm2d('syncbn', channels=1)
        nn.init.constant_(self._pixel_space_mask_batch_norm.weight, 0.1)

    def forward(self, mask_embeddings, class_embeddings, pixel_feature):
        _,_,N = mask_embeddings.shape
        B,_,Z,H,W = pixel_feature.shape
        pixel_space_feature = self._pixel_space_head_conv0bnact(pixel_feature)
        pixel_space_feature = self._pixel_space_head_conv1bnact(pixel_space_feature)
        pixel_space_feature = self._pixel_space_head_last_convbn(pixel_space_feature)
        pixel_space_normalized_feature = F.normalize(pixel_space_feature, p=2, dim=1)

        cluster_class_logits = self._transformer_class_head(class_embeddings).permute(0, 2, 1).contiguous()
        cluster_class_logits = add_bias_towards_void(cluster_class_logits)
        cluster_mask_kernel = self._transformer_mask_head(mask_embeddings)
        #print('pixel_space_normalized_feature:{},cluster_mask_kernel:{}'.format(pixel_space_normalized_feature.shape, cluster_mask_kernel.shape))
        mask_logits = torch.einsum('bczhw,bcn->bnzhw',
          pixel_space_normalized_feature, cluster_mask_kernel)
        mask_logits = mask_logits.view(B,N,Z*H*W)

        mask_logits = self._pixel_space_mask_batch_norm(mask_logits.unsqueeze(dim=1)).squeeze(dim=1)
        mask_logits = mask_logits.view(B,N,Z,H,W)

        class_wise_mask_logits = torch.einsum('bnzhw,bnk->bkzhw',
          mask_logits, cluster_class_logits)
       # print('class_wise_mask_logits:{}'.format(class_wise_mask_logits.shape))
        return {
            'class_logits': cluster_class_logits,
            'mask_logits': mask_logits,
            'pixel_feature': pixel_space_normalized_feature,
            'class_wise_mask_logits':class_wise_mask_logits}
       

class SIPL(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, epoch, backbone = 'swinunetr', num_queries = 32, 
                 drop_path_prob = 0.0,encoding = 'rand_embedding', num_samples=4, t_max=None):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.t_max = t_max
        self.epoch = epoch
        self.num_samples = num_samples
        self.backbone_name = backbone
        self.encoding = encoding
        #print('-------------num_classes:{}----------'.format(num_classes))
        if backbone == 'swinunetr':
            self.backbone = SwinUNETR(img_size=img_size,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        feature_size=48,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False,
                        )
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 48),
                nn.ReLU(inplace=True),
                nn.Conv3d(48, 8, kernel_size=1)
            )
        elif backbone == 'unet':
            self.backbone = UNet3D()
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 8, kernel_size=1)
            )
        elif backbone == 'unetr_pp':
            self.backbone = UNETR_PP(in_channels=1,
                             out_channels=out_channels,
                             feature_size=16,
                             num_heads=4,
                             depths=[3, 3, 3, 3],
                             dims=[32, 64, 128, 256],
                             do_ds=True,
                             )
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 16),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 8, kernel_size=1)
            )
        elif backbone == 'dints':
            ckpt = torch.load('./model/arch_code_cvpr.pth')
            node_a = ckpt["node_a"]
            arch_code_a = ckpt["arch_code_a"]
            arch_code_c = ckpt["arch_code_c"]

            dints_space = TopologyInstance(
                    channel_mul=1.0,
                    num_blocks=12,
                    num_depths=4,
                    use_downsample=True,
                    arch_code=[arch_code_a, arch_code_c]
                )

            self.backbone = DiNTS(
                    dints_space=dints_space,
                    in_channels=1,
                    num_classes=3,
                    use_downsample=True,
                    node_a=node_a,
                )
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 8, kernel_size=1)
            )
        elif backbone == 'unetpp':
            self.backbone = BasicUNetPlusPlus(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 8, kernel_size=1)
            )
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(backbone))

        if self.encoding == 'rand_embedding':
            self.organ_embedding = nn.Embedding(out_channels, 256)
        elif self.encoding == 'word_embedding':
            self.register_buffer('organ_embedding', torch.randn(out_channels, 512))
            self.text_to_vision = nn.Linear(512, 256)
        self.num_classes = out_channels

        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)
        weight_nums.append(8*8)
        weight_nums.append(8*1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.controller = nn.Conv3d(256+256, sum(weight_nums+bias_nums), kernel_size=1, stride=1, padding=0)

        self._num_queries = num_queries
        # learnable query features
        self._kmax_transformer_layers = nn.ModuleList()

        if backbone == 'unet':
            in_channels = [256, 128, 64]
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'swinunetr':
            in_channels = [384, 192, 96]
            # in_channels = [96]
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 768),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(768, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'unetr_pp':
            in_channels = [128, 64, 32]
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 256),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
            )
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(backbone)) 
        
        self._num_blocks = [2,2,2]
        self._cluster_centers = nn.Embedding(in_channels[0], num_queries)
        os2channels = {32: in_channels[0], 16: in_channels[1], 8:in_channels[2]}
        for index, output_stride in enumerate([32,16,8]):
            for _ in range(self._num_blocks[index]):
                self._kmax_transformer_layers.append(
                    kMaXTransformerLayer(num_classes=self.num_classes + 1,
                    in_channel_pixel=os2channels[output_stride],
                    in_channel_query=in_channels[0],
                    base_filters=int(in_channels[0]/2),
                    num_heads=8,
                    bottleneck_expansion=2,
                    key_expansion=1,
                    value_expansion=2,
                    drop_path_prob=drop_path_prob,
                    t_max=self.t_max)
                )
        self._class_embedding_projection = ConvBN1d(in_channels[0], in_channels[0], kernel_size=1, bias=False, norm='syncbn', 
                                                    act='gelu',conv_type='1d')
        self._mask_embedding_projection = ConvBN1d(in_channels[0], in_channels[0], kernel_size=1, bias=False, norm='syncbn', 
                                                    act='gelu',conv_type='1d')
        self._predcitor = kMaXPredictor(in_channel_pixel=in_channels[-1],\
            in_channel_query=in_channels[0], num_classes=self.num_classes + 1) 

    def load_params(self, model_dict):
        if self.backbone_name == 'swinunetr':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out' not in key:
                    store_dict[key] = model_dict[key]

            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')
              
    def get_class_specific_avg(self, feat, pred_logits):
        b,_,z,h,w = feat.shape
        pred_logits = F.interpolate(pred_logits, size=feat.shape[-3:], mode='trilinear', align_corners=True)
        #print('pred_logits shape:{}, feat shape:{}'.format(pred_logits.shape, feat.shape))
        pred_logits = F.softmax(pred_logits, dim = 1)
        pooled_centers = []
        for i in range(1, self.num_classes + 1):
            tmp_feat = feat * pred_logits[:,i,...].unsqueeze(1)
            tmp_pooled_feat = self.GAP(tmp_feat)
            pooled_centers.append(tmp_pooled_feat)

        pooled_centers = torch.stack(pooled_centers, dim = 1)

        return pooled_centers

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)
            # print(weight_splits[l].shape, bias_splits[l].shape)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x
            
    def forward(self, x_global, is_train=False):
        dec4, up_feats, out_unet = self.backbone(x_global) 
        B = up_feats[0].shape[0]
        cluster_centers = self._cluster_centers.weight.unsqueeze(0).repeat(B, 1, 1) # B x C x L

        if self.encoding == 'rand_embedding':
            task_encoding = self.organ_embedding.weight.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        elif self.encoding == 'word_embedding':
            task_encoding = F.relu(self.text_to_vision(self.organ_embedding))   
            task_encoding = task_encoding.unsqueeze(2).unsqueeze(2).unsqueeze(2)   # [13, 256, 1, 1, 1]
        
        current_transformer_idx = 0

        predictions_class = []
        predictions_mask = []
        predictions_pixel_feature = []
        class_wise_mask_logits = []

        for i, feat in enumerate(up_feats):
            for idx in range(self._num_blocks[i]):
                if idx == 0:
                    cluster_centers, prediction_result = self._kmax_transformer_layers[current_transformer_idx](
                        pixel_feature=feat, query_feature=cluster_centers, epoch=self.epoch, mask=None
                    )
                else:
                    cluster_centers, prediction_result = self._kmax_transformer_layers[current_transformer_idx](
                        pixel_feature=feat, query_feature=cluster_centers, epoch=self.epoch, mask=class_wise_mask_logits[-1]
                    )
                predictions_class.append(prediction_result['class_logits'])
                predictions_mask.append(prediction_result['mask_logits'])
                predictions_pixel_feature.append(prediction_result['pixel_feature'])
                class_wise_mask_logits.append(prediction_result['class_wise_mask_logits'])
                current_transformer_idx += 1
        
        class_embeddings = self._class_embedding_projection(cluster_centers)
        mask_embeddings = self._mask_embedding_projection(cluster_centers)

        prediction_result = self._predcitor(
            class_embeddings=class_embeddings,
            mask_embeddings=mask_embeddings,
            pixel_feature=up_feats[-1],
        )

        predictions_class.append(prediction_result['class_logits'])
        predictions_mask.append(prediction_result['mask_logits'])
        predictions_pixel_feature.append(prediction_result['pixel_feature'])  # [4, 128, 24, 24, 24]
        class_wise_mask_logits.append(prediction_result['class_wise_mask_logits'])  # [b,k,z,h,w]
        
        logits_array = []
        ## get the class specific-centers for each input
        class_pooled_features = self.get_class_specific_avg(dec4, class_wise_mask_logits[-1])
        
        for i in range(B):
            x_cond = torch.cat([class_pooled_features[i].repeat(1,1,1,1,1), task_encoding], 1)
            params = self.controller(x_cond)
            params.squeeze_(-1).squeeze_(-1).squeeze_(-1)
            head_inputs = self.precls_conv(out_unet[i].unsqueeze(0))
            head_inputs = head_inputs.repeat(self.num_classes,1,1,1,1)

            N, _, D, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, D, H, W)
            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)
            logits = self.heads_forward(head_inputs, weights, biases, N)
            logits_array.append(logits.reshape(1, -1, D, H, W))

        out_clip = torch.cat(logits_array, dim=0)
        if is_train:
            return out_clip, class_wise_mask_logits
        else:
            return out_clip