import torch
from torch import nn

# GCNet
class ContextBlock3d(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock3d, self).__init__()

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv3d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv3d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1, 1]),
                nn.ReLU6(inplace=True),  # yapf: disable
                nn.Conv3d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv3d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1, 1]),
                nn.ReLU6(inplace=True),  # yapf: disable
                nn.Conv3d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, time, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, T * H * W]
            input_x = input_x.view(batch, channel, time*height * width)
            # [N, 1, C, T * H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, T, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, T * H * W]
            context_mask = context_mask.view(batch, 1, time*height * width)
            # [N, 1, T * H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, T * H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1, 1]
            context = context.view(batch, channel, 1, 1, 1)
        else:
            # [N, C, 1, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

if __name__ == "__main__":

    in_tensor = torch.ones((10, 256, 10, 28, 28))

    cb = ContextBlock3d(inplanes=256, ratio=1./16.,pooling_type='avg')
    
    out_tensor = cb(in_tensor)

    print(in_tensor.shape)
    print(out_tensor.shape)