import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numbers
from doconv import DOConv2d


def conv(in_channels, out_channels, kernel_size, bias_attr=False, stride=1):
    return nn.Conv2D(in_channels,
                     out_channels,
                     kernel_size,
                     padding=(kernel_size // 2),
                     bias_attr=bias_attr,
                     stride=stride)


def doconv(in_channels, out_channels, kernel_size, bias_attr=False, stride=1):
    return DOConv2d(in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size // 2),
                    stride=stride,
                    bias=bias_attr)


def to_3d(x):
    b, c, h, w = x.shape
    x = paddle.reshape(x, [b, c, h * w])
    x = paddle.transpose(x, [0, 2, 1])
    return x


def to_4d(x, h, w):
    b, hw, c = x.shape
    x = paddle.reshape(x, [b, h, w, c])
    x = paddle.transpose(x, [0, 3, 1, 2])
    return x


def pixel_unshuffle(x, scale):
    """ Pixel unshuffle function.
    Args:
        x (paddle.Tensor): Input feature.
        scale (int): Downsample ratio.
    Returns:
        paddle.Tensor: the pixel unshuffled feature.
    """
    b, c, h, w = x.shape
    out_channel = c * (scale**2)
    assert h % scale == 0 and w % scale == 0
    hh = h // scale
    ww = w // scale
    x_reshaped = x.reshape([b, c, hh, scale, ww, scale])
    return x_reshaped.transpose([0, 1, 3, 5, 2,
                                 4]).reshape([b, out_channel, hh, ww])


## Channel Attention Layer
class CALayer(nn.Layer):
    def __init__(self, channel, reduction=16, bias_attr=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2D(channel,
                      channel // reduction,
                      1,
                      padding=0,
                      bias_attr=bias_attr), nn.ReLU(),
            nn.Conv2D(channel // reduction,
                      channel,
                      1,
                      padding=0,
                      bias_attr=bias_attr), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Channel Attention Block (CAB)
class CAB(nn.Layer):
    def __init__(self, n_feat, kernel_size, reduction, bias_attr, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(
            doconv(n_feat, n_feat, kernel_size, bias_attr=bias_attr))
        modules_body.append(act)
        modules_body.append(
            doconv(n_feat, n_feat, kernel_size, bias_attr=bias_attr))

        self.CA = CALayer(n_feat, reduction, bias_attr=bias_attr)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class CAFFT(nn.Layer):
    def __init__(self, n_feat, kernel_size, reduction, bias_attr, act):
        super(CAFFT, self).__init__()
        modules_body = []
        modules_body.append(ResDoFFT(n_feat))

        self.CA = CALayer(n_feat, reduction, bias_attr=bias_attr)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

##---------- Resizing Modules ----------
class DownSample(nn.Layer):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.Conv2D(in_channels,
                      in_channels + s_factor,
                      1,
                      stride=1,
                      padding=0,
                      bias_attr=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Layer):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2D(in_channels + s_factor,
                      in_channels,
                      1,
                      stride=1,
                      padding=0,
                      bias_attr=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Layer):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2D(in_channels + s_factor,
                      in_channels,
                      1,
                      stride=1,
                      padding=0,
                      bias_attr=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


##########################################################################
## U-Net
class Encoder(nn.Layer):
    def __init__(self, n_feat, kernel_size, reduction, act, bias_attr,
                 scale_unetfeats, csff):
        super(Encoder, self).__init__()
        
        self.encoder_level1 = [
            TransformerBlock(dim=n_feat, num_heads=4, ffn_expansion_factor=2.66, bias=bias_attr, LayerNorm_type='WithBias')
            for _ in range(1)]

        self.encoder_level2 = [
            TransformerBlock(dim=n_feat + scale_unetfeats, num_heads=4, ffn_expansion_factor=2.66, bias=bias_attr, LayerNorm_type='WithBias') 
            for _ in range(1)]

        self.encoder_level3 = [
            TransformerBlock(dim=n_feat + (scale_unetfeats * 2), num_heads=4, ffn_expansion_factor=2.66, bias=bias_attr, LayerNorm_type='WithBias') 
            for _ in range(1)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2D(n_feat,
                                       n_feat,
                                       kernel_size=1,
                                       bias_attr=bias_attr)
            self.csff_enc2 = nn.Conv2D(n_feat + scale_unetfeats,
                                       n_feat + scale_unetfeats,
                                       kernel_size=1,
                                       bias_attr=bias_attr)
            self.csff_enc3 = nn.Conv2D(n_feat + (scale_unetfeats * 2),
                                       n_feat + (scale_unetfeats * 2),
                                       kernel_size=1,
                                       bias_attr=bias_attr)

            self.csff_dec1 = nn.Conv2D(n_feat,
                                       n_feat,
                                       kernel_size=1,
                                       bias_attr=bias_attr)
            self.csff_dec2 = nn.Conv2D(n_feat + scale_unetfeats,
                                       n_feat + scale_unetfeats,
                                       kernel_size=1,
                                       bias_attr=bias_attr)
            self.csff_dec3 = nn.Conv2D(n_feat + (scale_unetfeats * 2),
                                       n_feat + (scale_unetfeats * 2),
                                       kernel_size=1,
                                       bias_attr=bias_attr)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(
                decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(
                decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(
                decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder(nn.Layer):
    def __init__(self, n_feat, kernel_size, reduction, act, bias_attr,
                 scale_unetfeats):
        super(Decoder, self).__init__()
        self.decoder_level1 = []
        self.decoder_level2 = []
        self.decoder_level3 = []

        for _ in range(1):
            self.decoder_level1.append(
                TransformerBlock(dim=n_feat, num_heads=4, ffn_expansion_factor=2.66, bias=bias_attr, LayerNorm_type='WithBias'))
            self.decoder_level2.append(
                TransformerBlock(dim=n_feat + scale_unetfeats, num_heads=4, ffn_expansion_factor=2.66, bias=bias_attr, LayerNorm_type='WithBias'))
            self.decoder_level3.append(
                TransformerBlock(dim=n_feat + (scale_unetfeats * 2), num_heads=4, ffn_expansion_factor=2.66, bias=bias_attr, LayerNorm_type='WithBias'))

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat,
                              kernel_size,
                              reduction,
                              bias_attr=bias_attr,
                              act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats,
                              kernel_size,
                              reduction,
                              bias_attr=bias_attr,
                              act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


## Original Resolution Block (ORB)
class ORB(nn.Layer):
    def __init__(self, n_feat, kernel_size, reduction, act, bias_attr, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [
            CAFFT(n_feat, kernel_size, reduction, bias_attr=bias_attr, act=act)
            for _ in range(num_cab)]

        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ORSNet(nn.Layer):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act,
                 bias_attr, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()
        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act,
                        bias_attr, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act,
                        bias_attr, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act,
                        bias_attr, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(
            UpSample(n_feat + scale_unetfeats, scale_unetfeats),
            UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(
            UpSample(n_feat + scale_unetfeats, scale_unetfeats),
            UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2D(n_feat,
                                   n_feat + scale_orsnetfeats,
                                   kernel_size=1,
                                   bias_attr=bias_attr)
        self.conv_enc2 = nn.Conv2D(n_feat,
                                   n_feat + scale_orsnetfeats,
                                   kernel_size=1,
                                   bias_attr=bias_attr)
        self.conv_enc3 = nn.Conv2D(n_feat,
                                   n_feat + scale_orsnetfeats,
                                   kernel_size=1,
                                   bias_attr=bias_attr)

        self.conv_dec1 = nn.Conv2D(n_feat,
                                   n_feat + scale_orsnetfeats,
                                   kernel_size=1,
                                   bias_attr=bias_attr)
        self.conv_dec2 = nn.Conv2D(n_feat,
                                   n_feat + scale_orsnetfeats,
                                   kernel_size=1,
                                   bias_attr=bias_attr)
        self.conv_dec3 = nn.Conv2D(n_feat,
                                   n_feat + scale_orsnetfeats,
                                   kernel_size=1,
                                   bias_attr=bias_attr)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(
            decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(
            self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(
            self.up_dec2(decoder_outs[2]))

        return x


# Supervised Attention Module
class SAM(nn.Layer):
    def __init__(self, n_feat, kernel_size, bias_attr):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias_attr=bias_attr)
        self.conv2 = conv(n_feat, 3, kernel_size, bias_attr=bias_attr)
        self.conv3 = conv(3, n_feat, kernel_size, bias_attr=bias_attr)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = F.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class BiasFree_LayerNorm(nn.Layer):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        # normalized_shape = [normalized_shape]

        assert len(normalized_shape) == 1

        # self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.weight = paddle.create_parameter(shape=normalized_shape,dtype='float32',
                                              default_initializer=nn.initializer.Constant(1.0))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / paddle.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Layer):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        # normalized_shape = normalized_shape.shape

        assert len(normalized_shape) == 1

        # self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.weight = paddle.create_parameter(shape=normalized_shape,dtype='float32',
                                              default_initializer=nn.initializer.Constant(1.0))
        # self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.bias = paddle.create_parameter(shape=normalized_shape,dtype='float32',
                                              default_initializer=nn.initializer.Constant(0.0))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / paddle.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Layer):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Layer):
    """ Gated-Dconv Feed-Forward Network (GDFN) """
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2D(dim, hidden_features*2, kernel_size=1, bias_attr=bias)

        self.dwconv = nn.Conv2D(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias_attr=bias)

        self.project_out = nn.Conv2D(hidden_features, dim, kernel_size=1, bias_attr=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, axis=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Layer):
    """ Multi-DConv Head Transposed Self-Attention (MDTA) """
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature = paddle.create_parameter(shape=[num_heads, 1, 1],dtype='float32',
                                              default_initializer=nn.initializer.Constant(1.0))

        self.qkv = nn.Conv2D(dim, dim*3, kernel_size=1, bias_attr=bias)
        self.qkv_dwconv = nn.Conv2D(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias_attr=bias)

        self.project_out = nn.Conv2D(dim, dim, kernel_size=1, bias_attr=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, axis=1)

        b1, hc, h1, w1 = q.shape
        # q = paddle.reshape(q, [b1, self.num_heads, -1, h1, w1])
        c = hc // self.num_heads
        q = paddle.reshape(q, [b1, self.num_heads, c, (h1*w1)])

        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        b1, hc, h1, w1 = k.shape
        # k = paddle.reshape(k, [b1, self.num_heads, -1, h1, w1])
        c = hc // self.num_heads
        k = paddle.reshape(k, [b1, self.num_heads, c, (h1*w1)])
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        b1, hc, h1, w1 = v.shape
        # v = paddle.reshape(v, [b1, self.num_heads, -1, h1, w1])
        c = hc // self.num_heads
        v = paddle.reshape(v, [b1, self.num_heads, c, (h1*w1)])
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = paddle.nn.functional.normalize(q, axis=-1)
        k = paddle.nn.functional.normalize(k, axis=-1)

        attn = (q @ k.transpose([0, 1, 3, 2])) * self.temperature
        attn = F.softmax(attn, axis=-1)

        out = (attn @ v)

        b, head, c, hw = out.shape

        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # out = paddle.reshape(out, [b, head, c, h, w])
        out = paddle.reshape(out, [b, head * c, h, w])

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Layer):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.do_fft = ResDoFFT(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        x = self.do_fft(x)
        return x


class ResDoFFT(nn.Layer):
    def __init__(self, out_channel, norm='backward'):
        super(ResDoFFT, self).__init__()
        self.main = nn.Sequential(
            doconv(out_channel, out_channel,kernel_size=3, stride=1),
            nn.ReLU(),
            doconv(out_channel, out_channel,kernel_size=3, stride=1))

        self.main_fft = nn.Sequential(
            doconv(2 * out_channel, 2 * out_channel, kernel_size=3, stride=1),
            nn.ReLU(),
            doconv(2 * out_channel, 2 * out_channel, kernel_size=3, stride=1))

        self.dim = out_channel
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = paddle.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag()
        y_real = y.real()
        y_f = paddle.concat([y_real, y_imag], axis=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = paddle.chunk(y, 2, axis=dim)
        y = paddle.complex(y_real, y_imag)
        y = paddle.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y


class MPRNet_mdta_v8(nn.Layer):
    def __init__(self,
                 in_c=3,
                 out_c=3,
                 n_feat=64,
                 scale_unetfeats=48,
                 scale_orsnetfeats=32,
                 num_cab=2,
                 kernel_size=3,
                 reduction=4,
                 bias_attr=False,
                 ffn_expansion_factor = 2.66,
                 LayerNorm_type = 'WithBias'):
        super(MPRNet_mdta_v8, self).__init__()
        act = nn.PReLU()
        self.padder_size = 8
        self.shallow_feat1 = nn.Sequential(
            doconv(in_c, n_feat, kernel_size, bias_attr=bias_attr),
            TransformerBlock(dim=n_feat, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias_attr, LayerNorm_type=LayerNorm_type))
        self.shallow_feat2 = nn.Sequential(
            doconv(in_c, n_feat, kernel_size, bias_attr=bias_attr),
            TransformerBlock(dim=n_feat, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias_attr, LayerNorm_type=LayerNorm_type))
        self.shallow_feat3 = nn.Sequential(
            doconv(in_c, n_feat, kernel_size, bias_attr=bias_attr),
            TransformerBlock(dim=n_feat, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias_attr, LayerNorm_type=LayerNorm_type))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat,
                                      kernel_size,
                                      reduction,
                                      act,
                                      bias_attr,
                                      scale_unetfeats,
                                      csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act,
                                      bias_attr, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat,
                                      kernel_size,
                                      reduction,
                                      act,
                                      bias_attr,
                                      scale_unetfeats,
                                      csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act,
                                      bias_attr, scale_unetfeats)

        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size,
                                    reduction, act, bias_attr, scale_unetfeats,
                                    num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias_attr=bias_attr)
        self.sam23 = SAM(n_feat, kernel_size=1, bias_attr=bias_attr)

        self.concat12 = doconv(n_feat * 2,
                             n_feat,
                             kernel_size,
                             bias_attr=bias_attr)
        self.concat23 = doconv(n_feat * 2,
                             n_feat + scale_orsnetfeats,
                             kernel_size,
                             bias_attr=bias_attr)
        self.tail = doconv(n_feat + scale_orsnetfeats,
                         out_c,
                         kernel_size,
                         bias_attr=bias_attr)


    def check_image_size(self, x):
        _, _, h, w = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


    def forward(self, x3_img):
        b, c, h_input, w_input = x3_img.shape
        x3_img = self.check_image_size(x3_img)

        # Original-resolution Image for Stage 3
        H = x3_img.shape[2]
        W = x3_img.shape[3]

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 2
        x2top_img = x3_img[:, :, 0:int(H / 2), :]
        x2bot_img = x3_img[:, :, int(H / 2):H, :]

        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:, :, :, 0:int(W / 2)]
        x1rtop_img = x2top_img[:, :, :, int(W / 2):W]
        x1lbot_img = x2bot_img[:, :, :, 0:int(W / 2)]
        x1rbot_img = x2bot_img[:, :, :, int(W / 2):W]

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)

        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)

        ## Concat deep features
        feat1_top = [
            paddle.concat((k, v), 3) for k, v in zip(feat1_ltop, feat1_rtop)
        ]
        feat1_bot = [
            paddle.concat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)
        ]

        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)

        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)

        ## Output image at Stage 1
        stage1_img = paddle.concat([stage1_img_top, stage1_img_bot], 2)
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top = self.shallow_feat2(x2top_img)
        x2bot = self.shallow_feat2(x2bot_img)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(paddle.concat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(paddle.concat([x2bot, x2bot_samfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        ## Concat deep features
        feat2 = [paddle.concat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)

        ## Apply SAM
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)

        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3 = self.shallow_feat3(x3_img)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(paddle.concat([x3, x3_samfeats], 1))

        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)

        stage3_img = self.tail(x3_cat)
        out = stage3_img + x3_img
        return [out[:, :, :h_input, :w_input], stage2_img[:, :, :h_input, :w_input], stage1_img[:, :, :h_input, :w_input]]


if __name__ == "__main__":
    img = paddle.randn(shape=[2, 3, 31, 31])
    Net = MPRNet_mdta_v8()
    output = Net(img)
    print(output[0].shape)