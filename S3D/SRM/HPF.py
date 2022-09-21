import numpy as np
import torch
import torch.nn as nn

# 30 SRM filtes
from SRM.srm_filter_kernel import all_normalized_hpf_list


# Pre-processing Module
class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        # Load 30 SRM Filters
        all_hpf_list_5x5 = []

        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

            all_hpf_list_5x5.append(hpf_item)

        all_hpf_list_5x5 = np.array(all_hpf_list_5x5).reshape(30, 1, 1, 5, 5)
        all_hpf_list_5x5 = np.concatenate([all_hpf_list_5x5 / 3] * 3, axis=1)

        hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 3, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv3d(3, 30, kernel_size=(1, 5, 5), padding=(0, 2, 2), bias=False)
        self.hpf.weight = hpf_weight

    def forward(self, input):

        output = self.hpf(input)
        #output = torch.clamp(output, min=-3, max=3)

        return output

if __name__ == '__main__':
    from torchsummary import summary
    
    model = HPF()
    summary(model, (3, 20, 224, 224), batch_size=32, device='cpu')
