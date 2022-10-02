import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, dim, time_size):
        super().__init__()
        time_padding = time_size // 2
        self.conv0 = nn.Conv3d(dim, dim, (time_size, 3, 3), padding=(time_padding, 1, 1), groups=dim)
        self.conv0_1 = nn.Conv3d(dim, dim, (time_size, 1, 5), padding=(time_padding, 0, 2), groups=dim)
        self.conv0_2 = nn.Conv3d(dim, dim, (time_size, 5, 1), padding=(time_padding, 2, 0), groups=dim)

        self.conv1_1 = nn.Conv3d(dim, dim, (time_size, 1, 7), padding=(time_padding, 0, 3), groups=dim)
        self.conv1_2 = nn.Conv3d(dim, dim, (time_size, 7, 1), padding=(time_padding, 3, 0), groups=dim)

        self.conv2_1 = nn.Conv3d(dim, dim, (time_size, 1, 11), padding=(time_padding, 0, 5), groups=dim)
        self.conv2_2 = nn.Conv3d(dim, dim, (time_size, 11, 1), padding=(time_padding, 5, 0), groups=dim)
        self.conv3 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model, time_size):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model, time_size)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

