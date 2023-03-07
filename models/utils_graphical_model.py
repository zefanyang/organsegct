#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/3/2021 9:11 PM
# @Author: yzf
import torch
import torch.nn as nn
from torch.nn import Conv1d, ReLU, Parameter

#=======================================UAGs===============================================
class UAG_RNN_4Neigh(nn.Module):
    """Four Neighbor Unidirectional Acyclic Graphs (UAGs). Henghui Ding et al., ICCV, 2019"""
    def __init__(self, in_dim):
        super(UAG_RNN_4Neigh, self).__init__()
        self.chanel_in = in_dim
        self.relu = ReLU()

        self.conv1 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv2 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # south
        self.conv4 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv5 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # east
        self.conv7 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv8 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # west
        self.conv9 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv10 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # north
        self.conv12 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv13 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # east
        self.conv15 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv16 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # west

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        ## s plane
        hs = x * 1
        for i in range(height):
            if i > 0:
                hs[:, :, i, :] = self.conv1(hs[:, :, i, :].clone()) + self.conv2(hs[:, :, i - 1, :].clone())  # why clone()?
                hs[:, :, i, :] = self.relu(hs[:, :, i, :].clone())

        ## e plane
        hse = hs * 1
        for j in range(width):
            if j > 0:
                hse[:, :, :, j] = self.conv4(hse[:, :, :, j].clone()) + self.conv5(hse[:, :, :, j - 1].clone())
            hse[:, :, :, j] = self.relu(hse[:, :, :, j].clone())

        ## w plane
        hsw = hs * 1
        for j in reversed(range(width)):
            if j < (width - 1):
                hsw[:, :, :, j] = self.conv7(hsw[:, :, :, j].clone()) + self.conv8(hsw[:, :, :, j + 1].clone())
            hsw[:, :, :, j] = self.relu(hsw[:, :, :, j].clone())

        ## n plane
        hn = x * 1
        for i in reversed(range(height)):
            if i < (height - 1):
                hn[:, :, i, :] = self.conv9(hn[:, :, i, :].clone()) + self.conv10(hn[:, :, i + 1, :].clone())
            hn[:, :, i, :] = self.relu(hn[:, :, i, :].clone())

        ## ne plane
        hne = hn * 1
        for j in range(width):
            if j > 0:
                hne[:, :, :, j] = self.conv12(hne[:, :, :, j].clone()) + self.conv13(hne[:, :, :, j - 1].clone())
            hne[:, :, :, j] = self.relu(hne[:, :, :, j].clone())

        ## nw plane
        hnw = hn * 1
        for j in reversed(range(width)):
            if j < (width - 1):
                hnw[:, :, :, j] = self.conv15(hnw[:, :, :, j].clone()) + self.conv16(hnw[:, :, :, j + 1].clone())
            hnw[:, :, :, j] = self.relu(hnw[:, :, :, j].clone())

        out = hse + hsw + hnw + hne

        return out

class UAG_RNN_8Neigh(nn.Module):
    """8 Neighbor Unidirectional Acyclic Graphs (UAGs). Henghui Ding et al., ICCV, 2019."""
    def     __init__(self, in_dim):
        super(UAG_RNN_8Neigh, self).__init__()
        self.chanel_in = in_dim
        self.relu = ReLU()

        self.gamma1 = Parameter(0.5 * torch.ones(1))
        self.gamma2 = Parameter(0.5 * torch.ones(1))
        self.gamma3 = Parameter(0.5 * torch.ones(1))
        self.gamma4 = Parameter(0.5 * torch.ones(1))
        self.conv1 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv2 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv3 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv4 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv5 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv6 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv7 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv8 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv9 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv10 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv11 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv12 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv13 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv14 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv15 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv16 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        ## s plane
        hs = x*1
        for i in range(height):
            if i>0:
                hs[:,:,i,:] = self.conv1(hs[:,:,i,:].clone()) + self.conv2(hs[:,:,i-1,:].clone())  # torch.clone() is differentiable
                hs[:,:,i,:] = self.relu(hs[:,:,i,:].clone())

        ## se plane
        # (i, j) = (i-1, j-1) diagonal + (i, j-1) left + (i-1, j) top
        hse = hs*1
        for j in range(width):
            if j>0:
                tmp = self.conv3(hse[:,:,:,j-1].clone())
                tmp = torch.cat((0*tmp[:,:,-1].view(m_batchsize, C, 1), tmp[:,:,0:-1]),2)  # diagonal. Place O first in height dimension
                hse[:,:,:,j] = self.conv4(hse[:,:,:,j].clone()) + self.conv5(hse[:,:,:,j-1].clone()) + self.gamma1*tmp
                del tmp
            hse[:,:,:,j] = self.relu(hse[:,:,:,j].clone())

        ## sw plane
        hsw = hs*1
        for j in reversed(range(width)):
            if j<(width-1):
                tmp = self.conv6(hsw[:,:,:,j+1].clone())
                tmp = torch.cat((0*tmp[:,:,-1].view(m_batchsize, C, 1), tmp[:,:,0:-1]),2)  # diagonal
                hsw[:,:,:,j] = self.conv7(hsw[:,:,:,j].clone()) + self.conv8(hsw[:,:,:,j+1].clone()) + self.gamma2*tmp
                del tmp
            hsw[:,:,:,j] = self.relu(hsw[:,:,:,j].clone())

        ## n plane
        hn = x*1
        for i in reversed(range(height)):
            if i<(height-1):
                hn[:,:,i,:] = self.conv9(hn[:,:,i,:].clone()) + self.conv10(hn[:,:,i+1,:].clone())
            hn[:,:,i,:] = self.relu(hn[:,:,i,:].clone())

        ## ne plane
        hne = hn*1
        for j in range(width):
            if j>0:
                tmp = self.conv11(hne[:,:,:,j-1].clone())
                tmp = torch.cat((tmp[:,:,1:], 0*tmp[:,:,0].view(m_batchsize, C, 1)),2)  # diagonal
                hne[:,:,:,j] = self.conv12(hne[:,:,:,j].clone()) + self.conv13(hne[:,:,:,j-1].clone()) + self.gamma3*tmp
                del tmp
            hne[:,:,:,j] = self.relu(hne[:,:,:,j].clone())

        ## nw plane
        hnw = hn*1
        for j in reversed(range(width)):
            if j<(width-1):
                tmp = self.conv14(hnw[:,:,:,j+1].clone())
                tmp = torch.cat((tmp[:,:,1:], 0*tmp[:,:,0].view(m_batchsize, C, 1)),2)  # diagonal
                hnw[:,:,:,j] = self.conv15(hnw[:,:,:,j].clone()) + self.conv16(hnw[:,:,:,j+1].clone()) + self.gamma4*tmp
                del tmp
            hnw[:,:,:,j] = self.relu(hnw[:,:,:,j].clone())

        out = hse + hsw + hnw + hne
        return out


#=======================================DAGs===============================================
class DAG_RNN_4Neigh(nn.Module):
    """ Directional Acyclic Graphs (DAGs)"""
    def __init__(self, in_dim):
        super(DAG_RNN, self).__init__()
        self.chanel_in = in_dim
        self.relu = ReLU()

        self.gamma1 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma2 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma4 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma5 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma7 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma8 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma10 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma11 = Parameter(torch.zeros(in_dim,in_dim))
    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        x_new = x.permute(1, 0, 2, 3)
        ## se plane
        hse = x_new*1
        for i in range(height):
            for j in range(width):
                if i>0:
                    hse[:,:,i,j] = hse[:,:,i,j].clone() + torch.mm(self.gamma1, hse[:,:,i-1,j].clone())
                if j>0:
                    hse[:,:,i,j] = hse[:,:,i,j].clone() + torch.mm(self.gamma2, hse[:,:,i,j-1].clone())
                hse[:,:,i,j] = self.relu(hse[:,:,i,j].clone())

        ## ne plane
        hne = x_new*1
        for i in reversed(range(height)):
            for j in range(width):
                if i<(height-1):
                    hne[:,:,i,j] = hne[:,:,i,j].clone() + torch.mm(self.gamma4, hne[:,:,i+1,j].clone())
                if j>0:
                    hne[:,:,i,j] = hne[:,:,i,j].clone() + torch.mm(self.gamma5, hne[:,:,i,j-1].clone())
                hne[:,:,i,j] = self.relu(hne[:,:,i,j].clone())

        ## nw plane
        hnw = x_new*1
        for i in reversed(range(height)):
            for j in reversed(range(width)):
                if i<(height-1):
                    hnw[:,:,i,j] = hnw[:,:,i,j].clone() + torch.mm(self.gamma7, hnw[:,:,i+1,j].clone())
                if j<(width-1):
                    hnw[:,:,i,j] = hnw[:,:,i,j].clone() + torch.mm(self.gamma8, hnw[:,:,i,j+1].clone())
                hnw[:,:,i,j] = self.relu(hnw[:,:,i,j].clone())

        ## sw plane
        hsw = x_new*1
        for i in range(height):
            for j in reversed(range(width)):
                if i>0:
                    hsw[:,:,i,j] = hsw[:,:,i,j].clone() + torch.mm(self.gamma10, hsw[:,:,i-1,j].clone())
                if j<(width-1):
                    hsw[:,:,i,j] = hsw[:,:,i,j].clone() + torch.mm(self.gamma11, hsw[:,:,i,j+1].clone())

        out = hse + hne + hnw + hsw
        out = out.permute(1, 0, 2, 3)
        return out





# class DAG_RNN_4Neigh(nn.Module):
#     """ Directional Acyclic Graphs (DAGs) [Bing Shuai et al. PAMI, 2018].
#     Matrix multiplication to convolution 1*1 kernel size"""
#     def __init__(self, in_dim):
#         super(DAG_RNN_4Neigh, self).__init__()
#         self.chanel_in = in_dim
#         self.relu = ReLU()
#
#         self.conv1 = Conv1d(in_dim, in_dim, kernel_size=1)
#         self.conv2 = Conv1d(in_dim, in_dim, kernel_size=1)
#         self.conv4 = Conv1d(in_dim, in_dim, kernel_size=1)
#         self.conv5 = Conv1d(in_dim, in_dim, kernel_size=1)
#         self.conv7 = Conv1d(in_dim, in_dim, kernel_size=1)
#         self.conv8 = Conv1d(in_dim, in_dim, kernel_size=1)
#         self.conv10 = Conv1d(in_dim, in_dim, kernel_size=1)
#         self.conv11 = Conv1d(in_dim, in_dim, kernel_size=1)
#
#     def forward(self,x):
#         m_batchsize, C, height, width = x.size()
#
#         ## se plane
#         hse = x
#         for i in range(height):
#             for j in range(width):
#                 if i>0:
#                     hse[:,:,i,j] = hse[:,:,i,j].clone() + self.conv1(hse[:,:,i-1,j:j+1].clone())
#                 if j>0:
#                     hse[:,:,i,j] = hse[:,:,i,j].clone() + self.conv2(hse[:,:,i:i+1,j-1].clone())
#                 hse[:,:,i,j] = self.relu(hse[:,:,i,j].clone())
#
#         ## ne plane
#         hne = x
#         for i in reversed(range(height)):
#             for j in range(width):
#                 if i<(height-1):
#                     hne[:,:,i,j] = hne[:,:,i,j].clone() + self.conv4(hne[:,:,i+1,j:j+1].clone())
#                 if j>0:
#                     hne[:,:,i,j] = hne[:,:,i,j].clone() + self.conv5(hne[:,:,i:i+1,j-1].clone())
#                 hne[:,:,i,j] = self.relu(hne[:,:,i,j].clone())
#
#         ## sw plane
#         hsw = x
#         for i in range(height):
#             for j in reversed(range(width)):
#                 if i > 0:
#                     hsw[:, :, i, j] = hsw[:, :, i, j].clone() + self.conv10(hsw[:, :, i - 1, j:j+1].clone())
#                 if j < (width - 1):
#                     hsw[:, :, i, j] = hsw[:, :, i, j].clone() + self.conv11(hsw[:, :, i:i+1, j + 1].clone())
#                 hsw[:, :, i, j] = self.relu(hsw[:, :, i, j].clone())
#
#         ## nw plane
#         hnw = x
#         for i in reversed(range(height)):
#             for j in reversed(range(width)):
#                 if i<(height-1):
#                     hnw[:,:,i,j] = hnw[:,:,i,j].clone() + self.conv7(hnw[:,:,i+1,j:j+1].clone())
#                 if j<(width-1):
#                     hnw[:,:,i,j] = hnw[:,:,i,j].clone() + self.conv8(hnw[:,:,i:i+1,j+1].clone())
#                 hnw[:,:,i,j] = self.relu(hnw[:,:,i,j].clone())
#
#         out = hse + hne + hsw + hnw
#         return out