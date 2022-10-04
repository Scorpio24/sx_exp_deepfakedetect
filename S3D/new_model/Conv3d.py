from torch import nn

class SepConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(SepConv3d, self).__init__()
        self.conv_s = nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size), stride=(1,stride,stride), padding=(0,padding,padding), bias=False)
        self.bn_s = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False)
        self.bn_t = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x

class DWSepConv3d(nn.Module):
    def __init__(self, dim, kernel_size:tuple, stride=1, padding:type=(0,0,0)):
        super(DWSepConv3d, self).__init__()
        self.conv_s = nn.Conv3d(dim, dim, kernel_size=(1,kernel_size[1],kernel_size[2]), stride=(1,stride,stride), padding=(0,padding[1],padding[2]), groups=dim, bias=False)
        # self.bn_s = nn.BatchNorm3d(dim, eps=1e-3, momentum=0.001, affine=True)
        # self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv3d(dim, dim, kernel_size=(kernel_size[0],1,1), stride=(stride,1,1), padding=(padding[0],0,0), groups=dim, bias=False)
        self.bn_t = nn.BatchNorm3d(dim, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        # x = self.bn_s(x)
        # x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x