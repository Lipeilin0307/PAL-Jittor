# -*- coding: utf-8 -*-
"""ISNet_No_Sigmoid 的 Jittor 迁移版 (CVPR'22, 梯度边缘分支 + TTOA 门控融合)。
源: PAL/model/ISNet/{ISNet_no_sigmoid,TTOA,dcn_v2,GatedSpatialConv,Resnet,mynn}.py。

迁移要点 (探针核对见 tools/probe_dcn_*.py):
1. DCNv2/DCN (dcn_v2.py): torchvision.ops.deform_conv2d 的手写纯 jittor 实现。
   实际配置 (TTOA.py L31-32): kernel (1,3)/(3,1), stride=1, padding (0,1)/(1,0),
   dilation=1, deformable_groups=1, 带 mask (sigmoid 后), 带 bias。
   **offset 通道布局经 numpy 双假设对拍确认为交错式: tap t 的 (y,x) = 通道 (2t, 2t+1)**
   (torchvision 语义; DCN wrapper 的 cat(o1,o2) 在此布局下逐位复现即可, 无需解读作者意图)。
   采样: 四角双线性加权, 角点出界贡献 0 (不做权重归一化), gather 用 x.reindex,
   autograd 经 reindex 散射回传 (与 torchvision fp64 梯度对拍 max diff ≤ 8.9e-15)。
   conv_offset_mask 按源码 init_offset() 零初始化 (PAL 从头训练, 此初始化是结构性的,
   保证 DCN 起步近似普通卷积)。
2. GatedSpatialConv2d 去 torch 私有基类 _ConvNd 继承: 显式持有 self.weight
   (out,in,1,1) 参数 + nn.conv2d 函数式调用 (bias=False 于全部实际使用点)。
   mynn.Norm2d == BatchNorm2d (config.py L70 cfg.MODEL.BNFUNC)。
3. register_buffer (GetGradientNoPadding.weight_v/h) -> 普通 jt.array 属性,
   同名进 state_dict, 键名 1:1。
4. 死模块忠实保留 (state_dict 全覆盖需要): SA/SA_att/cw/head2/conv2_1/conv16/
   res1-3/d1-3/gate1-3 均构造但 forward 不调用 (SA_att 的 17 通道 reshape 在数学上
   不可执行, 原版即如此, 属死代码)。
5. 输出契约 (train_model.py L285-292): forward 恒返回 (out, edge_out);
   out 为无 sigmoid logits, **edge_out 在内部已过 sigmoid**; 训练 loss 只算 out 分支
   (edge 分支 loss 在原版被注释), 指标/推理取 [0] 再过 sigmoid。
6. BN 全部 torch 默认 momentum=0.1 (jittor 同默认); Dropout(0.1) eval 恒等。
7. C=1 bias (dsn1/2/3、head.block[4]) 与进 interpolate 的张量按 acm/sct 惯例
   stop_fuse(), 规避 jittor 1.3.8.5 op0_outputstrideN 融合编译 bug, 数值不变。
8. ResidualBlock 的空 downsample Sequential 不能用真值判断 -> 显式 has_downsample 标志。
"""
import math

import jittor as jt
from jittor import nn


# --------------------------- DCNv2 手写实现 ---------------------------

def deform_conv2d(x, offset, weight, bias, stride=1, padding=(0, 0),
                  dilation=1, mask=None):
    """groups=1 版 torchvision deform_conv2d 纯 jittor 实现 (可反向)。

    offset 布局: tap t 的 (y_offset, x_offset) = 通道 (2t, 2t+1) (torchvision 交错语义,
    已经 numpy 对拍确认); mask 通道 = tap 序号。双线性采样, 角点出界贡献 0。
    """
    b, cin, h, w = x.shape
    cout, _, kh, kw = weight.shape
    sh = sw = stride
    ph, pw = padding
    dh = dw = dilation
    oh = (h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    ow = (w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
    kk = kh * kw

    dsize = (b, cin, oh, ow)
    bi = jt.arange(b).reshape(b, 1, 1, 1).broadcast(dsize)
    ci = jt.arange(cin).reshape(1, cin, 1, 1).broadcast(dsize)
    use64 = x.dtype == 'float64'

    def fcast(v):
        return v.float64() if use64 else v.float32()

    base_y = fcast(jt.arange(oh).reshape(1, 1, oh, 1)) * sh - ph
    base_x = fcast(jt.arange(ow).reshape(1, 1, 1, ow)) * sw - pw

    def gather(yi, xi):
        vic = fcast((yi >= 0) & (yi <= h - 1) & (xi >= 0) & (xi <= w - 1))
        yc = jt.clamp(yi, 0, h - 1).int32().broadcast(dsize)
        xc = jt.clamp(xi, 0, w - 1).int32().broadcast(dsize)
        return x.reindex([bi, ci, yc, xc]) * vic

    samples = []
    for t in range(kk):
        ky, kx = t // kw, t % kw
        ys = base_y + ky * dh + offset[:, 2 * t:2 * t + 1]      # (b,1,oh,ow)
        xs = base_x + kx * dw + offset[:, 2 * t + 1:2 * t + 2]
        y0, x0 = jt.floor(ys), jt.floor(xs)
        y1, x1 = y0 + 1.0, x0 + 1.0
        wy1, wy0 = ys - y0, y1 - ys
        wx1, wx0 = xs - x0, x1 - xs
        s = (gather(y0, x0) * wy0 * wx0 + gather(y0, x1) * wy0 * wx1 +
             gather(y1, x0) * wy1 * wx0 + gather(y1, x1) * wy1 * wx1)
        samples.append(s)                                       # (b,cin,oh,ow)

    S = jt.stack(samples, dim=2)                                # (b,cin,kk,oh,ow)
    if mask is not None:
        S = S * mask.reshape(b, 1, kk, oh, ow)
    # 权重累加: 广播乘 + sum, 不用 matmul (cublasGemmEx 不支持 fp64, CUDA 上会崩)
    W = weight.reshape(1, cout, cin, kk, 1, 1)
    out = (S.unsqueeze(1) * W).sum(dims=(2, 3))                 # (b,cout,oh,ow)
    if bias is not None:
        out = out + bias.reshape(1, cout, 1, 1)
    return out


class DCN(nn.Module):
    """dcn_v2.py 的 DCN (torchvision 封装版) jittor 重写。
    属性名与 torch 版一致: weight/bias (DCNv2 基类布局) + conv_offset_mask。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, deformable_groups=1):
        super().__init__()
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        ph, pw = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.kernel_size = (kh, kw)
        self.stride = stride
        self.padding = (ph, pw)
        self.dilation = dilation
        self.deformable_groups = deformable_groups

        stdv = 1.0 / math.sqrt(in_channels * kh * kw)
        w0 = jt.rand(out_channels, in_channels, kh, kw)  # U[0,1)
        self.weight = (w0 * 2 - 1) * stdv                # U(-stdv, stdv), 对齐 torch reset_parameters
        self.bias = jt.zeros(out_channels)

        channels_ = deformable_groups * 3 * kh * kw
        self.conv_offset_mask = nn.Conv2d(in_channels, channels_,
                                          kernel_size=self.kernel_size,
                                          stride=stride, padding=self.padding, bias=True)
        # 源码 init_offset(): 零初始化 (PAL 从头训练, 结构性初始化, 必须复刻)
        self.conv_offset_mask.weight.update(jt.zeros_like(self.conv_offset_mask.weight))
        self.conv_offset_mask.bias.update(jt.zeros_like(self.conv_offset_mask.bias))

    def execute(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = out.chunk(3, dim=1)
        offset = jt.concat([o1, o2], dim=1)
        mask = jt.sigmoid(mask)
        return deform_conv2d(x, offset, self.weight, self.bias,
                             stride=self.stride, padding=self.padding,
                             dilation=self.dilation, mask=mask)


# --------------------------- GatedSpatialConv (去 _ConvNd 继承) ---------------------------

class GatedSpatialConv2d(nn.Module):
    """torch 版继承私有 _ConvNd 仅为获得 weight 布局; 此处显式持有参数。
    实际使用点全部 kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False。
    _gate_conv: [BN(in+1), Conv1x1(in+1->in+1), ReLU, Conv1x1(in+1->1), BN(1), Sigmoid]
    (mynn.Norm2d == BatchNorm2d)。"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        # torch _ConvNd(padding_mode='zeros') + reset_parameters: xavier_normal_(weight)
        self.weight = jt.init.xavier_gauss((out_channels, in_channels, kh, kw))
        self.bias = jt.zeros(out_channels) if bias else None

        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels + 1),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # C=1 bias 融合 codegen bug 规避 (op0_outputstrideN, train 模式实测触发于
        # 该 bias 与 reduce.mean 的融合), 数值不变
        self._gate_conv[3].bias.stop_fuse()

    def execute(self, input_features, gating_features):
        alphas = self._gate_conv(jt.concat([input_features, gating_features], dim=1))
        input_features = input_features * (alphas + 1)
        return nn.conv2d(input_features, self.weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)


# --------------------------- vendor Resnet.BasicBlock / BasicBlock1 ---------------------------

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """Resnet.BasicBlock (mynn.Norm2d=BatchNorm2d); torch 版 __init__ 内的
    kaiming/BN 初始化遍历不迁移 (权重转换覆盖; 从头训练时框架默认 init 差异可接受,
    与 ACM vendor 版处理一致)。"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock1(nn.Module):
    """Resnet.BasicBlock1: 返回 (out, out1), out1 为不含残差的 relu(conv 路径)。"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = residual + out1
        out1 = self.relu(out1)
        out = self.relu(out)
        return out, out1


# --------------------------- TTOA ---------------------------

class TTOA(nn.Module):
    def __init__(self, low_channels, high_channels, c_kernel=3, r_kernel=3,
                 use_att=False, use_process=True):
        super(TTOA, self).__init__()
        self.l_c = low_channels
        self.h_c = high_channels
        self.c_k = c_kernel
        self.r_k = r_kernel
        self.att = use_att
        if self.l_c != self.h_c:
            raise ValueError('Low and Hih channels need to be the same!')
        self.dcn_row = DCN(self.l_c, self.h_c, kernel_size=(1, self.r_k),
                           stride=1, padding=(0, self.r_k // 2))
        self.dcn_colum = DCN(self.l_c, self.h_c, kernel_size=(self.c_k, 1),
                             stride=1, padding=(self.c_k // 2, 0))
        self.sigmoid = nn.Sigmoid()
        self.csa = nn.Conv2d(self.l_c, self.h_c, 1, 1, 0) if use_att else None
        if use_process:
            self.preprocess = nn.Sequential(
                nn.Conv2d(self.l_c, self.h_c // 2, 1, 1, 0),
                nn.Conv2d(self.h_c // 2, self.l_c, 1, 1, 0))
        else:
            self.preprocess = None

    def execute(self, a_low, a_high):
        if self.preprocess is not None:
            a_low = self.preprocess(a_low)
            a_high = self.preprocess(a_high)

        a_low_c = self.dcn_colum(a_low)
        a_low_cw = self.sigmoid(a_low_c)
        a_low_cw = a_low_cw * a_high
        a_colum = a_low + a_low_cw

        a_low_r = self.dcn_row(a_low)
        a_low_rw = self.sigmoid(a_low_r)
        a_low_rw = a_low_rw * a_high
        a_row = a_low + a_low_rw

        if self.csa is not None:
            a_TTOA = self.csa(a_row + a_colum)
        else:
            a_TTOA = a_row + a_colum
        return a_TTOA


# --------------------------- ISNet 主体 ---------------------------

class GetGradientNoPadding(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        # torch register_buffer -> state_dict 包含; jittor 普通 Var 属性同名注册
        self.weight_v = jt.array(kernel_v, dtype='float32').reshape(1, 1, 3, 3)
        self.weight_h = jt.array(kernel_h, dtype='float32').reshape(1, 1, 3, 3)

    def execute(self, x):
        def _grad(z):
            zv = nn.conv2d(z, self.weight_v, None, 1, 1)
            zh = nn.conv2d(z, self.weight_h, None, 1, 1)
            return jt.sqrt(zv * zv + zh * zh + 1e-6)

        return jt.concat([_grad(x[:, 0:1]), _grad(x[:, 1:2]), _grad(x[:, 2:3])], dim=1)


class TFD(nn.Module):
    def __init__(self, inch, outch):
        super(TFD, self).__init__()
        self.res1 = BasicBlock1(inch, outch, stride=1, downsample=None)
        self.res2 = BasicBlock1(inch, outch, stride=1, downsample=None)
        self.gate = GatedSpatialConv2d(inch, outch)

    def execute(self, x, f_x):
        u_0 = x
        u_1, delta_u_0 = self.res1(u_0)
        _, u_2 = self.res2(u_1)
        u_3_pre = self.gate(u_2, f_x)
        u_3 = 3 * delta_u_0 + u_2 + u_3_pre
        return u_3


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.has_downsample = bool(downsample)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

    def execute(self, x):
        residual = x
        x = self.body(x)
        if self.has_downsample:
            residual = self.downsample(residual)
        return nn.relu(x + residual)


class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),   # 0
            nn.BatchNorm2d(inter_channels),                                 # 1
            nn.ReLU(),                                                      # 2
            nn.Dropout(0.1),                                                # 3
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)                # 4
        )
        if out_channels == 1:
            # jittor 1.3.8.5 融合 codegen bug 规避 (同 acm.py): C=1 bias stop_fuse
            self.block[4].bias.stop_fuse()

    def execute(self, x):
        return self.block(x)


class sa_layer(nn.Module):
    """Channel Spatial Group module (ISNet 中 SA/SA_att 为死模块, 仅保留参数)。"""
    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = jt.zeros(1, channel // (2 * groups), 1, 1)
        self.cbias = jt.ones(1, channel // (2 * groups), 1, 1)
        self.sweight = jt.zeros(1, channel // (2 * groups), 1, 1)
        self.sbias = jt.ones(1, channel // (2 * groups), 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        return x.reshape(b, -1, h, w)

    def execute(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)
        out = jt.concat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)
        return self.channel_shuffle(out, 2)


class ISNet_No_Sigmoid(nn.Module):
    def __init__(self, layer_blocks=[4] * 3, channels=[8, 16, 32, 64]):
        super(ISNet_No_Sigmoid, self).__init__()

        stem_width = int(channels[0])
        self.stem = nn.Sequential(
            nn.BatchNorm2d(3),                                                # 0
            nn.Conv2d(3, stem_width, 3, 2, 1, bias=False),                    # 1
            nn.BatchNorm2d(stem_width),                                       # 2
            nn.ReLU(),                                                        # 3
            nn.Conv2d(stem_width, stem_width, 3, 1, 1, bias=False),           # 4
            nn.BatchNorm2d(stem_width),                                       # 5
            nn.ReLU(),                                                        # 6
            nn.Conv2d(stem_width, 2 * stem_width, 3, 1, 1, bias=False),       # 7
            nn.BatchNorm2d(2 * stem_width),                                   # 8
            nn.ReLU(),                                                        # 9
            nn.MaxPool2d(3, 2, 1),                                            # 10
        )
        self.TTOA_low = TTOA(channels[2], channels[2])
        self.TTOA_high = TTOA(channels[1], channels[1])
        self.layer1 = self._make_layer(ResidualBlock, layer_blocks[0],
                                       channels[1], channels[1], 1)
        self.layer2 = self._make_layer(ResidualBlock, layer_blocks[1],
                                       channels[1], channels[2], 2)
        self.layer3 = self._make_layer(ResidualBlock, layer_blocks[2],
                                       channels[2], channels[3], 2)

        self.deconv2 = nn.ConvTranspose2d(channels[3], channels[2], 4, 2, 1)
        self.uplayer2 = self._make_layer(ResidualBlock, layer_blocks[1],
                                         channels[2], channels[2], 1)
        self.deconv1 = nn.ConvTranspose2d(channels[2], channels[1], 4, 2, 1)
        self.uplayer1 = self._make_layer(ResidualBlock, layer_blocks[0],
                                         channels[1], channels[1], 1)

        self.head = _FCNHead(channels[1], 1)
        # edge branch
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(32, 1, 1)
        self.dsn3 = nn.Conv2d(16, 1, 1)

        # ===== 死模块 (forward 未调用, 为 state_dict 键名 1:1 而保留) =====
        self.res1 = BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1)
        self.res2 = BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.res3 = BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.fuse = nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=False)
        self.cw = nn.Conv2d(4, 1, kernel_size=1, padding=0, bias=False)
        self.gate1 = GatedSpatialConv2d(32, 32)
        self.gate2 = GatedSpatialConv2d(16, 16)
        self.gate3 = GatedSpatialConv2d(8, 8)
        self.sigmoid = nn.Sigmoid()
        self.SA = sa_layer(4, 2)
        self.SA_att = sa_layer(17, 2)
        self.dsup = nn.Conv2d(3, 64, 1)
        self.head2 = _FCNHead(channels[1], 3)
        self.conv2_1 = nn.Conv2d(3, 1, 1)
        self.conv16 = nn.Conv2d(3, 16, 1)
        self.myb1 = TFD(64, 64)
        self.myb2 = TFD(64, 64)
        self.myb3 = TFD(64, 64)
        self.grad_extractor = GetGradientNoPadding()

        # C=1 bias 融合 bug 规避 (acm/sct 惯例)
        for m in (self.dsn1, self.dsn2, self.dsn3):
            m.bias.stop_fuse()

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        layer = []
        downsample = (in_channels != out_channels) or (stride != 1)
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)

    def execute(self, x):
        hei, wid = x.shape[2], x.shape[3]
        x_grad = self.grad_extractor(x)

        x1 = self.stem(x)
        c1 = self.layer1(x1)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)

        deconc2 = self.deconv2(c3)
        fusec2 = self.TTOA_low(deconc2, c2)
        upc2 = self.uplayer2(fusec2)

        deconc1 = self.deconv1(upc2)
        fusec1 = self.TTOA_high(deconc1, c1)
        upc1 = self.uplayer1(fusec1)

        # stop_fuse: C=1 输出进 interpolate 前打断融合边界, 数值不变
        s1 = nn.interpolate(self.dsn1(c3).stop_fuse(), size=(hei, wid),
                            mode='bilinear', align_corners=True)
        s2 = nn.interpolate(self.dsn2(upc2).stop_fuse(), size=(hei, wid),
                            mode='bilinear', align_corners=True)
        s3 = nn.interpolate(self.dsn3(upc1).stop_fuse(), size=(hei, wid),
                            mode='bilinear', align_corners=True)

        m1f = nn.interpolate(x_grad, size=(hei, wid), mode='bilinear',
                             align_corners=True)
        m1f = self.dsup(m1f)
        cs1 = self.myb1(m1f, s1)
        cs2 = self.myb2(cs1, s2)
        cs = self.myb3(cs2, s3)
        cs = self.fuse(cs)
        cs = nn.interpolate(cs.stop_fuse(), (hei, wid), mode='bilinear',
                            align_corners=True)
        edge_out = self.sigmoid(cs)

        upc1 = nn.interpolate(upc1, size=(hei, wid), mode='bilinear')
        fuse = edge_out * upc1 + upc1

        pred = self.head(fuse)

        out = nn.interpolate(pred.stop_fuse(), size=(hei, wid), mode='bilinear')
        return out, edge_out


if __name__ == '__main__':
    jt.flags.use_cuda = 1
    net = ISNet_No_Sigmoid()
    inputs = jt.rand(1, 3, 256, 256)
    out, edge = net(inputs)
    print('out:', out.shape, 'edge_out:', edge.shape)
    print('Params = %.2f M' % (sum(p.numel() for p in net.parameters()) / 1e6))
