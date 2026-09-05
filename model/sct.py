# -*- coding: utf-8 -*-
"""SCTransNet_No_Sigmoid 的 Jittor 迁移版。
源: PAL/model/SCTransNet/SCTransNet_no_sigmoid.py (torch, 679 行, TGRS'24 UNet+channel-cross transformer)。

迁移要点(全部经 tools/probe_sct_*.py 探针核对):
1. einops.rearrange -> reshape/permute 手写:
   - 'b (head c) h w -> b head c (h w)' (num_attention_heads=1): reshape(b, 1, c, h*w)
   - 'b c (h w) -> b c h w': reshape(b, c, h, w)
   - 'b c h w -> b (h w) c' (to_3d): permute(0,2,3,1).reshape(b, h*w, c)
   - 'b (h w) c -> b c h w' (to_4d): reshape(b, h, w, c).permute(0,3,1,2)
2. ml_collections.ConfigDict -> 普通嵌套 dict, 数值按源码 get_CTranS_config 默认
   (KV_size=480, num_heads=4, num_layers=4, patch_sizes=[16,8,4,2], base_channel=32)。
3. nn.InstanceNorm2d(1) (psi, 作用于 (b,1,c,hw) 注意力图):
   torch 默认 affine=False/track_running_stats=False -> jt.nn.InstanceNorm2d(1, affine=False)。
   注意 jittor 默认 affine=True, 必须显式关闭; 否则 state_dict 多出 weight/bias, 键覆盖校验失败。
   探针实测 affine=False 时 jittor IN 无参数, train/eval 均用实例统计, 与 torch 一致。
4. 手写 LayerNorm (BiasFree/WithBias, var(unbiased=False), eps=1e-5):
   用 (x-mu) 二阶矩手算有偏方差, 避免 jt.var 的 unbiased 语义差异。
5. thop import 仅 __main__ 用 -> 删。
6. nn.Upsample(scale_factor=2) (UpBlock_attention.up) torch 默认 mode='nearest',
   探针确认 jittor nn.Upsample 默认一致; Reconstruct 的 Upsample(mode='bilinear')
   默认 align_corners=False; 深监督 gt2..5 的 interpolate(mode='bilinear', align_corners=True)
   已探针逐点核对。
7. F.normalize(dim=-1) -> x / clamp(||x||_2, min=1e-12) 手写, 探针一致。
8. F.avg_pool2d 全窗 / AdaptiveAvgPool2d(1) -> mean(dims=(2,3), keepdims=True) 等价。
9. 输出契约: mode='train' 且 deepsuper 时返回 (gt5, gt4, gt3, gt2, d0, out) 共 6 个 logits
   (均无 sigmoid); 否则仅返回 out。train_model.py L294-304 对 6 分支各算一次 edgeSCE 再取 mean,
   辅助分支 targets 用 max_pool2d/interpolate-nearest 下采样(该逻辑在训练脚本侧, 不在本文件)。
10. 死参数保留: Channel_Embeddings.position_embeddings (定义未用) 与 channel_attn 的
    16 个 q*_attn* 标量参数均在 torch state_dict 中, 为保持键名 1:1 全覆盖, 原样保留。
11. Block_ViT x4 由 copy.deepcopy 自同一实例 -> 4 层共享初始化值(源行为, 保持)。
12. 构造签名与原版完全一致; train_model.py L609-614 以 mode='train' 构造。
13. jittor 1.3.8.5 融合 codegen bug 规避(同 acm.py): C=1 的 gt_conv 输出接 interpolate
    融合时可能生成未定义标识符 op0_outputstrideN, 对进入 interpolate 的张量 stop_fuse()
    打断融合边界, 数值不变。
"""
import copy
import math

import jittor as jt
from jittor import nn


def get_CTranS_config():
    """普通 dict 版配置, 数值与源码 ml_collections 默认完全一致。"""
    config = {
        'KV_size': 480,  # KV_size = Q1 + Q2 + Q3 + Q4
        'patch_sizes': [16, 8, 4, 2],
        'base_channel': 32,  # base channel of U-Net
        'n_classes': 1,
        'transformer': {
            'num_heads': 4,
            'num_layers': 4,
            # ********** useless (前向未使用, 仅保持配置等价) **********
            'embeddings_dropout_rate': 0.1,
            'attention_dropout_rate': 0.1,
            'dropout_rate': 0,
        },
    }
    return config


class Channel_Embeddings(nn.Module):
    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        patch_size = (patchsize, patchsize) if isinstance(patchsize, int) else tuple(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = nn.Conv2d(in_channels, in_channels,
                                          kernel_size=patch_size, stride=patch_size)
        # 死参数: 源码定义但 forward 未使用; 保留以维持 state_dict 键名 1:1
        self.position_embeddings = jt.zeros(1, n_patches, in_channels)
        self.dropout = nn.Dropout(config['transformer']['embeddings_dropout_rate'])

    def execute(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        return x


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.scale_factor = scale_factor

    def execute(self, x):
        if x is None:
            return None
        # torch nn.Upsample(scale_factor=sf, mode='bilinear') 默认 align_corners=False;
        # jittor interpolate 不接受元组 scale_factor, 两个维度相同取标量
        sf = self.scale_factor[0] if isinstance(self.scale_factor, (tuple, list)) else self.scale_factor
        x = nn.interpolate(x, scale_factor=sf, mode='bilinear')
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


# spatial-embedded Single-head Channel-cross Attention (SSCA)
class Attention_org(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = config['KV_size']
        self.channel_num = channel_num
        self.num_attention_heads = 1
        # torch 默认 affine=False/track_running_stats=False; jittor 默认 affine=True, 必须关掉
        self.psi = nn.InstanceNorm2d(self.num_attention_heads, affine=False)
        self.softmax = nn.Softmax(dim=3)

        self.mhead1 = nn.Conv2d(channel_num[0], channel_num[0] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead2 = nn.Conv2d(channel_num[1], channel_num[1] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead3 = nn.Conv2d(channel_num[2], channel_num[2] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead4 = nn.Conv2d(channel_num[3], channel_num[3] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadk = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadv = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)

        self.q1 = nn.Conv2d(channel_num[0] * self.num_attention_heads, channel_num[0] * self.num_attention_heads,
                            kernel_size=3, stride=1, padding=1,
                            groups=channel_num[0] * self.num_attention_heads // 2, bias=False)
        self.q2 = nn.Conv2d(channel_num[1] * self.num_attention_heads, channel_num[1] * self.num_attention_heads,
                            kernel_size=3, stride=1, padding=1,
                            groups=channel_num[1] * self.num_attention_heads // 2, bias=False)
        self.q3 = nn.Conv2d(channel_num[2] * self.num_attention_heads, channel_num[2] * self.num_attention_heads,
                            kernel_size=3, stride=1, padding=1,
                            groups=channel_num[2] * self.num_attention_heads // 2, bias=False)
        self.q4 = nn.Conv2d(channel_num[3] * self.num_attention_heads, channel_num[3] * self.num_attention_heads,
                            kernel_size=3, stride=1, padding=1,
                            groups=channel_num[3] * self.num_attention_heads // 2, bias=False)
        self.k = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads,
                           kernel_size=3, stride=1, padding=1,
                           groups=self.KV_size * self.num_attention_heads, bias=False)
        self.v = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads,
                           kernel_size=3, stride=1, padding=1,
                           groups=self.KV_size * self.num_attention_heads, bias=False)

        self.project_out1 = nn.Conv2d(channel_num[0], channel_num[0], kernel_size=1, bias=False)
        self.project_out2 = nn.Conv2d(channel_num[1], channel_num[1], kernel_size=1, bias=False)
        self.project_out3 = nn.Conv2d(channel_num[2], channel_num[2], kernel_size=1, bias=False)
        self.project_out4 = nn.Conv2d(channel_num[3], channel_num[3], kernel_size=1, bias=False)

        # ****************** useless (源码死参数, 保留以维持 state_dict 键名 1:1) ******
        self.q1_attn1 = jt.array([0.2])
        self.q1_attn2 = jt.array([0.2])
        self.q1_attn3 = jt.array([0.2])
        self.q1_attn4 = jt.array([0.2])

        self.q2_attn1 = jt.array([0.2])
        self.q2_attn2 = jt.array([0.2])
        self.q2_attn3 = jt.array([0.2])
        self.q2_attn4 = jt.array([0.2])

        self.q3_attn1 = jt.array([0.2])
        self.q3_attn2 = jt.array([0.2])
        self.q3_attn3 = jt.array([0.2])
        self.q3_attn4 = jt.array([0.2])

        self.q4_attn1 = jt.array([0.2])
        self.q4_attn2 = jt.array([0.2])
        self.q4_attn3 = jt.array([0.2])
        self.q4_attn4 = jt.array([0.2])

    def _normalize(self, x):
        # torch F.normalize(dim=-1): x / max(||x||_2, eps=1e-12)
        norm = jt.sqrt((x * x).sum(-1, keepdims=True))
        return x / jt.clamp(norm, min_v=1e-12)

    def execute(self, emb1, emb2, emb3, emb4, emb_all):
        b, c, h, w = emb1.shape
        q1 = self.q1(self.mhead1(emb1))
        q2 = self.q2(self.mhead2(emb2))
        q3 = self.q3(self.mhead3(emb3))
        q4 = self.q4(self.mhead4(emb4))
        k = self.k(self.mheadk(emb_all))
        v = self.v(self.mheadv(emb_all))

        # rearrange 'b (head c) h w -> b head c (h w)', head=1
        q1 = q1.reshape(b, 1, q1.shape[1], h * w)
        q2 = q2.reshape(b, 1, q2.shape[1], h * w)
        q3 = q3.reshape(b, 1, q3.shape[1], h * w)
        q4 = q4.reshape(b, 1, q4.shape[1], h * w)
        k = k.reshape(b, 1, k.shape[1], h * w)
        v = v.reshape(b, 1, v.shape[1], h * w)

        q1 = self._normalize(q1)
        q2 = self._normalize(q2)
        q3 = self._normalize(q3)
        q4 = self._normalize(q4)
        k = self._normalize(k)

        attn1 = jt.matmul(q1, k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn2 = jt.matmul(q2, k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn3 = jt.matmul(q3, k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn4 = jt.matmul(q4, k.transpose(-2, -1)) / math.sqrt(self.KV_size)

        attention_probs1 = self.softmax(self.psi(attn1))
        attention_probs2 = self.softmax(self.psi(attn2))
        attention_probs3 = self.softmax(self.psi(attn3))
        attention_probs4 = self.softmax(self.psi(attn4))

        out1 = jt.matmul(attention_probs1, v)
        out2 = jt.matmul(attention_probs2, v)
        out3 = jt.matmul(attention_probs3, v)
        out4 = jt.matmul(attention_probs4, v)

        out_1 = out1.mean(1)
        out_2 = out2.mean(1)
        out_3 = out3.mean(1)
        out_4 = out4.mean(1)

        # rearrange 'b c (h w) -> b c h w'
        out_1 = out_1.reshape(b, out_1.shape[1], h, w)
        out_2 = out_2.reshape(b, out_2.shape[1], h, w)
        out_3 = out_3.reshape(b, out_3.shape[1], h, w)
        out_4 = out_4.reshape(b, out_4.shape[1], h, w)

        O1 = self.project_out1(out_1)
        O2 = self.project_out2(out_2)
        O3 = self.project_out3(out_3)
        O4 = self.project_out4(out_4)
        weights = None

        return O1, O2, O3, O4, weights


def to_3d(x):
    # rearrange 'b c h w -> b (h w) c'
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def to_4d(x, h, w):
    # rearrange 'b (h w) c -> b c h w'
    b = x.shape[0]
    c = x.shape[2]
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        assert len(normalized_shape) == 1
        self.weight = jt.ones(normalized_shape)
        self.normalized_shape = normalized_shape

    def execute(self, x):
        # 有偏方差 (unbiased=False), eps=1e-5
        mu = x.mean(-1, keepdims=True)
        xc = x - mu
        sigma = (xc * xc).mean(-1, keepdims=True)
        return x / jt.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        assert len(normalized_shape) == 1
        self.weight = jt.ones(normalized_shape)
        self.bias = jt.zeros(normalized_shape)
        self.normalized_shape = normalized_shape

    def execute(self, x):
        mu = x.mean(-1, keepdims=True)
        xc = x - mu
        sigma = (xc * xc).mean(-1, keepdims=True)  # var(unbiased=False)
        return xc / jt.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def execute(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class eca_layer_2d(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer_2d, self).__init__()
        padding = k_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=k_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
        self.k_size = k_size

    def execute(self, x):
        b, c = x.shape[0], x.shape[1]
        out = self.avg_pool(x)
        out = out.reshape(b, 1, c)
        out = self.conv(out)
        out = out.reshape(b, c, 1, 1)
        return out * x


# Complementary Feed-forward Network (CFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)
        self.eca = eca_layer_2d(dim)

    def execute(self, x):
        x_3, x_5 = self.project_in(x).chunk(2, dim=1)
        x1_3 = self.relu3(self.dwconv3x3(x_3))
        x1_5 = self.relu5(self.dwconv5x5(x_5))
        x = jt.concat([x1_3, x1_5], dim=1)
        x = self.project_out(x)
        x = self.eca(x)
        return x


#  Spatial-channel Cross Transformer Block (SCTB)
class Block_ViT(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT, self).__init__()
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(config['KV_size'], LayerNorm_type='WithBias')

        self.channel_attn = Attention_org(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.ffn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.ffn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.ffn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

        self.ffn1 = FeedForward(channel_num[0], ffn_expansion_factor=2.66, bias=False)
        self.ffn2 = FeedForward(channel_num[1], ffn_expansion_factor=2.66, bias=False)
        self.ffn3 = FeedForward(channel_num[2], ffn_expansion_factor=2.66, bias=False)
        self.ffn4 = FeedForward(channel_num[3], ffn_expansion_factor=2.66, bias=False)

    def execute(self, emb1, emb2, emb3, emb4):
        embcat = []
        for e in (emb1, emb2, emb3, emb4):
            if e is not None:
                embcat.append(e)
        emb_all = jt.concat(embcat, dim=1)
        org1, org2, org3, org4 = emb1, emb2, emb3, emb4

        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)
        cx1, cx2, cx3, cx4, weights = self.channel_attn(cx1, cx2, cx3, cx4, emb_all)
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        cx4 = org4 + cx4 if emb4 is not None else None

        org1, org2, org3, org4 = cx1, cx2, cx3, cx4
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        x4 = self.ffn_norm4(cx4) if emb4 is not None else None
        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None
        x4 = self.ffn4(x4) if emb4 is not None else None
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        x4 = x4 + org4 if emb4 is not None else None

        return x1, x2, x3, x4, weights


class Encoder(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.encoder_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.encoder_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.encoder_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        for _ in range(config['transformer']['num_layers']):
            layer = Block_ViT(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def execute(self, emb1, emb2, emb3, emb4):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, emb3, emb4, weights = layer_block(emb1, emb2, emb3, emb4)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        return emb1, emb2, emb3, emb4, attn_weights


class ChannelTransformer(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128, 256, 512], patchSize=[32, 16, 8, 4]):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        self.embeddings_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config, self.patchSize_2, img_size=img_size // 2, in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config, self.patchSize_3, img_size=img_size // 4, in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(config, self.patchSize_4, img_size=img_size // 8, in_channels=channel_num[3])
        self.encoder = Encoder(config, vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1, scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1, scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1, scale_factor=(self.patchSize_3, self.patchSize_3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1, scale_factor=(self.patchSize_4, self.patchSize_4))

    def execute(self, en1, en2, en3, en4):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        encoded1, encoded2, encoded3, encoded4, attn_weights = self.encoder(emb1, emb2, emb3, emb4)

        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3) if en3 is not None else None
        x4 = self.reconstruct_4(encoded4) if en4 is not None else None

        x1 = x1 + en1 if en1 is not None else None
        x2 = x2 + en2 if en2 is not None else None
        x3 = x3 + en3 if en3 is not None else None
        x4 = x4 + en4 if en4 is not None else None

        return x1, x2, x3, x4, attn_weights


def get_activation(activation_type):
    # 注意: 不能照搬 torch 版 "hasattr(nn, name.lower())" 的写法 ——
    # jittor 的 nn.relu 是函数而非 Module 类, 直接实例化会报错。
    act = activation_type.lower()
    if act == 'relu':
        return nn.ReLU()
    cls = getattr(nn, activation_type, None)
    if isinstance(cls, type) and issubclass(cls, nn.Module):
        return cls()
    return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = [CBN(in_channels, out_channels, activation)]
    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def execute(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def execute(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def execute(self, x):
        return x.reshape(x.shape[0], -1)


class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU()

    def execute(self, g, x):
        h_x, w_x = x.shape[2], x.shape[3]
        avg_pool_x = x.mean(dims=(2, 3), keepdims=True)  # == F.avg_pool2d(x, (h,w), stride=(h,w))
        channel_att_x = self.mlp_x(avg_pool_x)
        h_g, w_g = g.shape[2], g.shape[3]
        avg_pool_g = g.mean(dims=(2, 3), keepdims=True)
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        b, c = x.shape[0], x.shape[1]
        scale = jt.sigmoid(channel_att_sum).reshape(b, c, 1, 1)  # unsqueeze+expand_as 由广播等价
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        # torch nn.Upsample(scale_factor=2) 默认 mode='nearest'
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def execute(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = jt.concat([skip_x_att, up], dim=1)
        return self.nConvs(x)


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def execute(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class SCTransNet_No_Sigmoid(nn.Module):
    def __init__(self, config=None, n_channels=3, n_classes=1, img_size=256, vis=False, mode='test', deepsuper=True):
        super().__init__()
        if config is None:
            config = get_CTranS_config()
        self.vis = vis
        self.deepsuper = deepsuper
        print('Deep-Supervision:', deepsuper)
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config['base_channel']  # basic channel 32
        block = Res_block
        self.pool = nn.MaxPool2d(2, 2)
        self.inc = self._make_layer(block, n_channels, in_channels)
        self.down_encoder1 = self._make_layer(block, in_channels, in_channels * 2, 1)
        self.down_encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 1)
        self.down_encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)
        self.down_encoder4 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)
        self.mtc = ChannelTransformer(config, vis, img_size,
                                      channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
                                      patchSize=config['patch_sizes'])
        self.up_decoder4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up_decoder3 = UpBlock_attention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up_decoder2 = UpBlock_attention(in_channels * 4, in_channels, nb_Conv=2)
        self.up_decoder1 = UpBlock_attention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=1, stride=1)

        if self.deepsuper:
            self.gt_conv5 = nn.Sequential(nn.Conv2d(in_channels * 8, 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(in_channels * 4, 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(in_channels * 1, 1, 1))
            self.outconv = nn.Conv2d(5 * 1, 1, 1)

        # jittor 1.3.8.5 融合 codegen bug 规避(同 acm.py _FCNHead): C=1 的 conv bias
        # 形状为 [1], 其 array->broadcast_to->add 融合内核会引用未定义标识符
        # op0_outputstrideN 导致编译失败(train 模式实测触发)。对 bias 标记 stop_fuse
        # 使该参数不进入融合内核, 数值不变。
        if n_classes == 1:
            self.outc.bias.stop_fuse()
        if self.deepsuper:
            for m in (self.gt_conv5, self.gt_conv4, self.gt_conv3, self.gt_conv2):
                m[0].bias.stop_fuse()
            self.outconv.bias.stop_fuse()

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = [block(input_channels, output_channels)]
        for _ in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def execute(self, x):
        x1 = self.inc(x)                    # 32  256 256
        x2 = self.down_encoder1(self.pool(x1))   # 64  128 128
        x3 = self.down_encoder2(self.pool(x2))   # 128 64  64
        x4 = self.down_encoder3(self.pool(x3))   # 256 32  32
        d5 = self.down_encoder4(self.pool(x4))   # 256 16  16
        #  CCT
        f1, f2, f3, f4 = x1, x2, x3, x4
        x1, x2, x3, x4, att_weights = self.mtc(x1, x2, x3, x4)
        x1 = x1 + f1
        x2 = x2 + f2
        x3 = x3 + f3
        x4 = x4 + f4
        #  Feature fusion
        d4 = self.up_decoder4(d5, x4)
        d3 = self.up_decoder3(d4, x3)
        d2 = self.up_decoder2(d3, x2)
        out = self.outc(self.up_decoder1(d2, x1))
        # deep supervision
        if self.deepsuper:
            gt_5 = self.gt_conv5(d5)
            gt_4 = self.gt_conv4(d4)
            gt_3 = self.gt_conv3(d3)
            gt_2 = self.gt_conv2(d2)
            # jittor 1.3.8.5 融合 codegen bug 规避(同 acm.py): C=1 输出接 interpolate
            # 融合时可能引用未定义标识符 op0_outputstrideN, stop_fuse() 打断融合, 数值不变
            gt5 = nn.interpolate(gt_5.stop_fuse(), scale_factor=16, mode='bilinear', align_corners=True)
            gt4 = nn.interpolate(gt_4.stop_fuse(), scale_factor=8, mode='bilinear', align_corners=True)
            gt3 = nn.interpolate(gt_3.stop_fuse(), scale_factor=4, mode='bilinear', align_corners=True)
            gt2 = nn.interpolate(gt_2.stop_fuse(), scale_factor=2, mode='bilinear', align_corners=True)
            d0 = self.outconv(jt.concat((gt2, gt3, gt4, gt5, out), 1))

            if self.mode == 'train':
                return (gt5, gt4, gt3, gt2, d0, out)
            else:
                return out
        else:
            return out


if __name__ == '__main__':
    jt.flags.use_cuda = 1
    model = SCTransNet_No_Sigmoid(mode='train', deepsuper=True)
    inputs = jt.rand(1, 3, 256, 256)
    outputs = model(inputs)
    print('outputs:', [tuple(o.shape) for o in outputs])
    n_params = sum(p.numel() for p in model.parameters())
    print('Params = %.2f M' % (n_params / 1000 ** 2))
