import torch
from torch import nn
from torch.autograd import Variable

from matplotlib import pyplot as plt
import numpy as np


class CosDistanceMatcher(nn.Module):
    def __init__(self, stride=1, eps=1e-6, inf=-1e9):
        super().__init__()
        self.stride = stride
        self.eps = eps
        self.inf = inf

    def forward(self, x, filter, patch_norms, mask):
        conv_output = torch.nn.functional.conv2d(x, filter.contiguous(), bias=None, stride=self.stride, padding=0, dilation=1, groups=1)
        normed_output = conv_output / (patch_norms + self.eps)
        mask = Variable(torch.FloatTensor(mask[..., :-filter.shape[2]+1, :-filter.shape[3]+1]).cuda())
        normed_output[mask==1] = self.inf
        
        indexes_list = []
        while len(indexes_list) != 4:
            normed_output, indexes = torch.max(normed_output, 0)
            indexes_list.append(indexes)
            
        result_index = [indexes_list[-1].data.cpu().numpy()[0]]
        result_index.append(indexes_list[-2][result_index[-1]].data.cpu().numpy()[0])
        result_index.append(indexes_list[-3][result_index[-1], result_index[-2]].data.cpu().numpy()[0])
        result_index.append(indexes_list[-4][result_index[-1], result_index[-2], result_index[-3]].data.cpu().numpy()[0])
        
        return result_index[::-1]



class ShiftLayer(nn.Module):
    def __init__(self, stride=1, eps=1e-6, kernel_size=4, inf=-1e9):
        super().__init__()
        self.stride = stride
        self.eps = eps
        self.matcher = CosDistanceMatcher(stride, eps, inf)
        self.kernel_size = kernel_size

    def forward(self, low_level_features, hight_level_features, mask):
        kernel_size = self.kernel_size
        
        patch_norms = Variable(torch.zeros(low_level_features.shape[0],
                                  1,
                                  low_level_features.shape[2] - kernel_size + 1,
                                  low_level_features.shape[3] - kernel_size + 1),
                              ).cuda()
        
        result = torch.zeros_like(low_level_features)
        filter_counts = torch.zeros_like(low_level_features)
        
        for i in range(low_level_features.shape[2] - kernel_size + 1):
            for j in range(low_level_features.shape[3] - kernel_size + 1):
                patch_norms[0, 0, i, j] = torch.norm(low_level_features[0, :, i:i+kernel_size,j:j+kernel_size], 2)
        
        for i in range(0, low_level_features.shape[2] - kernel_size + 1, self.stride):
            for j in range(0, low_level_features.shape[3] - kernel_size + 1, self.stride):
                if mask[i, j] == 0 or mask[i, j+kernel_size] == 0 or mask[i+kernel_size, j] == 0 or mask[i+kernel_size, j+kernel_size] == 0:
                    continue
                filter = hight_level_features[0, :, i:i+kernel_size,j:j+kernel_size][None, :]
                index_list = self.matcher(low_level_features, filter, patch_norms, mask)
                result_filter = low_level_features[index_list[0], :,
                                                   index_list[2]:index_list[2]+kernel_size,
                                                   index_list[3]:index_list[3]+kernel_size]
                
                result[0, :, i:i+kernel_size,j:j+kernel_size] = result[0, :, i:i+kernel_size,j:j+kernel_size] + result_filter
                filter_counts[0, :, i:i+kernel_size,j:j+kernel_size] = filter_counts[0, :, i:i+kernel_size,j:j+kernel_size] + 1
        
        result = result / (filter_counts + self.eps)
        result = result * (filter_counts != 0).float() + low_level_features * (filter_counts == 0).float()
        
        return result
    
    
class ShiftNet(nn.Module):
    def __init__(self, stride=1, eps=1e-6, kernel_size=4, inf=-1e9):
        super().__init__()
        self.first32_output = None
        self.second32_output = None
        
        self.conv256 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
        self.conv128 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv64 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.conv32 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.conv16 = nn.Conv2d(512, 512, 4, stride=2, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(512, 512, 4, stride=2, padding=1)
        
        self.up_conv2 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        self.up_conv4 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.up_conv8 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.up_conv16 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.up_conv32 = nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1)
        
        self.shift = ShiftLayer()
        
        self.up_conv64 = nn.ConvTranspose2d(768, 128, 4, stride=2, padding=1)
        self.up_conv128 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        self.up_conv256 = nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)
        
        self.LReLU = nn.LeakyReLU(0.2)
        self.ReLu = nn.ReLU()
        self.tanh = nn.Tanh()   

    def forward(self, img, mask):
        self.first32_output = None
        self.second32_output = None
        
        x128 = self.conv256(img)
        x64 = self.conv128(self.LReLU(x128))
        x32 = self.conv64(self.LReLU(x64))
        x16 = self.conv32(self.LReLU(x32))
        x8 = self.conv16(self.LReLU(x16))
        x4 = self.conv8(self.LReLU(x8))
        x2 = self.conv4(self.LReLU(x4))
        x1 = self.conv2(self.LReLU(x2))
        
        up_x2 = self.up_conv2(self.ReLu(x1))
        up_x2 = torch.cat([x2, up_x2], dim=1)
        
        up_x4 = self.up_conv4(self.ReLu(up_x2))
        up_x4 = torch.cat([x4, up_x4], dim=1)
        
        up_x8 = self.up_conv8(self.ReLu(up_x4))
        up_x8 = torch.cat([x8, up_x8], dim=1)
        
        up_x16 = self.up_conv16(self.ReLu(up_x8))
        up_x16 = torch.cat([x16, up_x16], dim=1)
        
        up_x32 = self.up_conv32(self.ReLu(up_x16))
        self.first32_output = x32
        self.second32_output = up_x32
        shift = self.shift(x32, up_x32, mask)
        up_x32 = torch.cat([x32, up_x32, shift], dim=1)
        
        up_x64 = self.up_conv64(self.ReLu(up_x32))
        up_x64 = torch.cat([x64, up_x64], dim=1)
        
        up_x128 = self.up_conv128(self.ReLu(up_x64))
        up_x128 = torch.cat([x128, up_x128], dim=1)
        
        up_x256 = self.up_conv256(self.ReLu(up_x128))
        up_x256 = self.tanh(up_x256)

        
        return up_x256