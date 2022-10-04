import torch
import torch.nn as nn
from new_model.msca_3d import SpatialAttention
from new_model.Conv3d import BasicConv3d
from new_model.Conv3d import SepConv3d
from new_model.Conv3d import DWSepConv3d


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

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

class InceptionMixer(nn.Module):
    def __init__(self, input_channels, tran_ratio, time_size):
        super().__init__()
        tran_chans = make_divisible(input_channels * tran_ratio, 32)
        conv_chans = input_channels - tran_chans
        self.high = conv_chans
        self.low  = tran_chans

        self.maxpool_fc = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            BasicConv3d(self.high // 2, self.high // 2, 1),
        )

        self.fc_dw = nn.Sequential(
            BasicConv3d(self.high // 2, self.high // 2, 1),
            DWSepConv3d(self.high // 2, (3,3,3), padding=(1,1,1)),
            #nn.Conv3d(self.high // 2, self.high // 2, 3, padding=1, groups=self.high // 2),
            nn.BatchNorm3d(self.high // 2, eps=1e-3, momentum=0.001, affine=True),
        )
        
        self.attn = SpatialAttention(self.low, time_size)

    def forward(self, x):

        X_h1 = x[:, :self.high//2, ...]
        X_h2 = x[:, self.high//2:self.high, ...]
        X_l  = x[:, -self.low:, ...]

        Y_h1 = self.maxpool_fc(X_h1)
        Y_h2 = self.fc_dw(X_h2)

        Y_l = self.attn(X_l)

        Y_c = torch.cat([Y_l, Y_h1, Y_h2], dim=1)
    
        return Y_c

class iFormerBlock(nn.Module):
    def __init__(
        self, input_channels, tran_ratio, time_size,
        mlp_ratio=4.,  drop=0.,
        act_layer=nn.GELU, 
        ):
        super().__init__()
        dim = input_channels
        self.norm1 = nn.BatchNorm3d(dim, eps=1e-3, momentum=0.001, affine=True)
        self.inceptionmixer = InceptionMixer(input_channels, tran_ratio, time_size)
        self.norm2 = nn.BatchNorm3d(dim, eps=1e-3, momentum=0.001, affine=True)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):

        x = x + self.inceptionmixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x

if __name__ == "__main__":
    from torchsummary import summary
    from torch.utils.tensorboard import SummaryWriter
    
    model = iFormerBlock(192, 0.5, 3)
    #summary(model, (192, 10, 56, 56), batch_size=11, device='cpu')
    
    tb_writer = SummaryWriter(log_dir="runs/model_test")
    init_img = torch.zeros((11, 192, 10, 56, 56))
    tb_writer.add_graph(model, init_img)
    tb_writer.close()