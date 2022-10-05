import torch.nn as nn
from new_model.Conv3d import DWSepConv3d

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = DWSepConv3d(dim, (3,3,3), 1, (1,1,1))

    def forward(self, x):
        x = self.dwconv(x)
        return x

# SegNeXt
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class AttentionModule(nn.Module):
    def __init__(self, dim, time_size):
        super().__init__()
        time_padding = time_size // 2

        self.conv0 = DWSepConv3d(dim, (time_size, 3, 3), padding=(time_padding, 1, 1))

        self.conv0_1 = DWSepConv3d(dim, (time_size, 5, 5), padding=(time_padding, 2, 2))
       
        self.conv1_1 = DWSepConv3d(dim, (time_size, 7, 7), padding=(time_padding, 3, 3))
        
        # self.conv2_1 = DWSepConv3d(dim, (time_size, 11, 11), padding=(time_padding, 5, 5))

        self.conv3 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        # attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        # attn_1 = self.conv1_2(attn_1)

        # attn_2 = self.conv2_1(attn)
        # attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 #+ attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model, time_size):
        super().__init__()
        # self.d_model = d_model
        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model, time_size)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x):
        # shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        # x = x + shorcut
        return x

class MSCAN_half(nn.Module):
    def __init__(
        self, input_channels, time_size
        ):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(input_channels, eps=1e-3, momentum=0.001, affine=True)
        self.attn = SpatialAttention(input_channels, time_size)

    def forward(self, x):

        x = x + self.attn(self.norm1(x))
        
        return x

class MSCAN(nn.Module):
    def __init__(
        self, input_channels, time_size,
        mlp_ratio=4.,  drop=0.,
        act_layer=nn.GELU,
        ):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(input_channels, eps=1e-3, momentum=0.001, affine=True)
        self.attn = SpatialAttention(input_channels, time_size)
        self.norm2 = nn.BatchNorm3d(input_channels, eps=1e-3, momentum=0.001, affine=True)
        mlp_hidden_dim = int(input_channels * mlp_ratio)
        self.mlp = Mlp(in_features=input_channels, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x):

        x = x + self.attn(self.norm1(x))
        # x = x + self.mlp(self.norm2(x))
        x = self.mlp(self.norm2(x))
        
        return x
