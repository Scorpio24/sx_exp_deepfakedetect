from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F

from SRM.HPF import HPF
from SRM.HPF import HPF_3

from new_model.Conv3d import SepConv3d
from new_model.Conv3d import SepConv3dV2
from new_model.Conv3d import BasicConv3d
from new_model.iformer_3d import iFormerBlock
from new_model.iformer_3d import iFormerBlock_light
from new_model.msca_3d import MSCAN_half
from new_model.msca_3d import MSCAN

class msca_S3D_SRM(nn.Module):
    def __init__(self, num_class, SRM_net):
        super(msca_S3D_SRM, self).__init__()

        self.SRM_net = SRM_net
        if SRM_net == 'yes':
            input_channels = 3
        else:
            input_channels = 3

        self.SRM = HPF_3()
        self.base = nn.Sequential( # input:bs*(3/30) *20*224*224
            SepConv3d(input_channels, 64, kernel_size=7, stride=2, padding=3),#out:bs*64*10*112*112
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),#out:bs*64*10*56*56
            MSCAN_half(64, 1),#out:bs*64*10*56*56

            BasicConv3d(64, 64, kernel_size=1, stride=1),#out:bs*64*10*56*56
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),#out:bs*192*10*56*56
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),#out:bs*192*10*28*28
            iFormerBlock_light(192, 1/4, 1),#out:bs*192*10*28*28
            iFormerBlock(192, 1/4, 1),#out:bs*192*10*28*28

            BasicConv3d(192, 320, kernel_size=1, stride=1),#out:bs*320*10*28*28
            # SepConv3d(128, 160, kernel_size=3, stride=1, padding=1),#out:bs*160*10*28*28
            nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),#out:bs*320*5*14*14
            iFormerBlock_light(320, 1/3, 3),#out:bs*320*10*28*28
            iFormerBlock_light(320, 1/3, 3),#out:bs*320*10*28*28
            iFormerBlock(320, 1/3, 3),#out:bs*320*10*28*28
            iFormerBlock_light(320, 1/2, 3),#out:bs*320*10*28*28
            iFormerBlock_light(320, 1/2, 3),#out:bs*320*10*28*28
            iFormerBlock(320, 1/2, 3),#out:bs*320*10*28*28
            iFormerBlock_light(320, 2/3, 3),#out:bs*320*10*28*28
            iFormerBlock_light(320, 2/3, 3),#out:bs*320*10*28*28
            iFormerBlock(320, 2/3, 3),#out:bs*320*10*28*28

            # BasicConv3d(320, 384, kernel_size=1, stride=1),#out:bs*512*5*14*14
            # SepConv3d(160, 256, kernel_size=3, stride=1, padding=1),#out:bs*256*5*14*14
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0)),#out:bs*320*2*7*7
            Mixed_5b(320),#out:bs*512*2*7*7
            Mixed_5c(512)#out:bs*1024*2*7*7
            
        )
        self.fc = nn.Sequential(nn.Conv3d(1024, num_class, kernel_size=1, stride=1, bias=True),)

    def forward(self, x):
        if self.SRM_net == 'yes':
            y = x + self.SRM(x)
        else:
            y = x
        y = self.base(y)
        y = F.avg_pool3d(y, (2, y.size(3), y.size(4)), stride=1)
        y = self.fc(y)
        y = y.view(y.size(0), y.size(1), y.size(2))
        logits = torch.mean(y, 2)

        return logits

class msca_S3D(nn.Module):
    def __init__(self, num_class, SRM_net):
        super(msca_S3D, self).__init__()

        self.SRM_net = SRM_net
        if SRM_net == 'yes':
            input_channels = 30
        else:
            input_channels = 3

        self.SRM = HPF()
        self.base = nn.Sequential( # input:bs*(3/30) *20*224*224
            SepConv3d(input_channels, 64, kernel_size=7, stride=2, padding=3),#out:bs*64*10*112*112
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),#out:bs*64*10*56*56
            # MSCAN(64, 1),#out:bs*64*10*56*56

            BasicConv3d(64, 64, kernel_size=1, stride=1),#out:bs*64*10*56*56
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),#out:bs*128*10*56*56
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),#out:bs*128*10*28*28
            iFormerBlock(192, 1/4, 1),#out:bs*128*10*28*28
            iFormerBlock(192, 1/4, 1),#out:bs*128*10*28*28

            BasicConv3d(192, 320, kernel_size=1, stride=1),#out:bs*320*10*28*28
            # SepConv3d(128, 160, kernel_size=3, stride=1, padding=1),#out:bs*160*10*28*28
            nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),#out:bs*160*5*14*14
            # iFormerBlock_light(320, 1/3, 3),#out:bs*128*10*28*28
            iFormerBlock(320, 1/3, 3),#out:bs*128*10*28*28
            iFormerBlock(320, 1/3, 3),#out:bs*128*10*28*28
            # iFormerBlock_light(320, 1/2, 3),#out:bs*128*10*28*28
            iFormerBlock(320, 1/2, 3),#out:bs*128*10*28*28
            iFormerBlock(320, 1/2, 3),#out:bs*128*10*28*28
            # iFormerBlock_light(320, 2/3, 3),#out:bs*128*10*28*28
            iFormerBlock(320, 2/3, 3),#out:bs*128*10*28*28
            iFormerBlock(320, 2/3, 3),#out:bs*128*10*28*28

            # BasicConv3d(320, 384, kernel_size=1, stride=1),#out:bs*512*5*14*14
            # SepConv3d(160, 256, kernel_size=3, stride=1, padding=1),#out:bs*256*5*14*14
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0)),#out:bs*256*2*7*7
            Mixed_5b(320),#out:bs*512*2*7*7
            Mixed_5c(512)#out:bs*1024*2*7*7
            
        )
        self.fc = nn.Sequential(nn.Conv3d(1024, num_class, kernel_size=1, stride=1, bias=True),)

    def forward(self, x):
        if self.SRM_net == 'yes':
            y = self.SRM(x)
        else:
            y = x
        y = self.base(y)
        y = F.avg_pool3d(y, (2, y.size(3), y.size(4)), stride=1)
        y = self.fc(y)
        y = y.view(y.size(0), y.size(1), y.size(2))
        logits = torch.mean(y, 2)

        return logits
class Mixed_5b(nn.Module):
    def __init__(self, input_channels):
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(input_channels, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(input_channels, 96, kernel_size=1, stride=1),
            SepConv3dV2(96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(input_channels, 16, kernel_size=1, stride=1),
            SepConv3dV2(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(input_channels, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Mixed_5c(nn.Module):
    def __init__(self, input_channels):
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(input_channels, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(input_channels, 192, kernel_size=1, stride=1),
            SepConv3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(input_channels, 48, kernel_size=1, stride=1),
            SepConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(input_channels, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class testmodel(nn.Module):
    def __init__(self) -> None:
        super(testmodel, self).__init__()
        self.conv1 = BasicConv3d(30, 64, 3,1,1)
        self.conv2 = SepConv3d(30, 64,3,1,1)
    
    def forward(self, x):
        # y = self.conv1(x)
        y = self.conv2(x)
        return y

if __name__ == '__main__':
    from torchsummary import summary
    from thop import profile, clever_format
    
    model = msca_S3D(1, 'yes')
    #summary(model, (3, 20, 224, 224), batch_size=11, device='cpu')

    # model = testmodel()
    input = torch.randn(10, 3, 20, 224, 224)
    flops, params = profile(model, inputs=(input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)