import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        #self.dropout = DropBlock2D(block_size=7, keep_prob=0.9)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
    def forward(self, x):
        out = self.conv1(x)
        #out = self.dropout(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

#Method [17]
class Method_of_23(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision = False):
        super().__init__()

        nb_filter = [16, 32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值
        self.drop_first = nn.Dropout(p = 0.025)
        self.drop = nn.Dropout(p=0.05)
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.trans3_1 = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], 2, stride=2, padding=0)
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[3], nb_filter[3], nb_filter[3])

        self.trans2_2 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], 2, stride=2, padding=0)
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[2], nb_filter[2], nb_filter[2])

        self.trans1_3 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], 2, stride=2, padding=0)
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[1], nb_filter[1], nb_filter[1])

        self.trans0_4 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], 2, stride=2, padding=0)
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[0], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.drop_first(self.pool(x1_0)))
        x3_0 = self.conv3_0(self.drop(self.pool(x2_0)))
        x4_0 = self.conv4_0(self.drop(self.pool(x3_0)))
        x3_1 = self.conv3_1(self.drop(torch.cat([x3_0, self.trans3_1(x4_0)], 1)))
        x2_2 = self.conv2_2(self.drop(torch.cat([x2_0, self.trans2_2(x3_1)], 1)))
        x1_3 = self.conv1_3(self.drop(torch.cat([x1_0, self.trans1_3(x2_2)], 1)))
        x0_4 = self.conv0_4(self.drop(torch.cat([x0_0, self.trans0_4(x1_3)], 1)))
        output = self.final(x0_4)
        return output

#Unet
class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3,  is_train=True, deep_supervision = False):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        #self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv2_1 = VGGBlock(nb_filter[3]+nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        #self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
       x0_0 = self.conv0_0(input)
       x1_0 = self.conv1_0(self.pool(x0_0))
       x2_0 = self.conv2_0(self.pool(x1_0))
       x3_0 = self.conv3_0(self.pool(x2_0))
       x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
       x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1)], 1))
       x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))
       output = self.final(x0_3)
       return output

#Unet++
class UNetplusplusL3(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
       # print('input:',input.shape)
        x0_0 = self.conv0_0(input)
       # print('x0_0:',x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
       # print('x1_0:',x1_0.shape)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
       # print('x0_1:',x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        #print('x2_0:',x2_0.shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        #print('x1_1:',x1_1.shape)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        #print('x0_2:',x0_2.shape)

        x3_0 = self.conv3_0(self.pool(x2_0))
        #print('x3_0:',x3_0.shape)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        #print('x2_1:',x2_1.shape)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        #print('x1_2:',x1_2.shape)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        #print('x0_3:',x0_3.shape)
       # x4_0 = self.conv4_0(self.pool(x3_0))
        #print('x4_0:',x4_0.shape)
        #x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        #print('x3_1:',x3_1.shape)
        #x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        #print('x2_2:',x2_2.shape)
       # x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        #print('x1_3:',x1_3.shape)
        #x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        #print('x0_4:',x0_4.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
           # output4 = self.final4(x0_4)
            b = [output1, output2, output3]
            return [output1, output2, output3]

        else:
            output = self.final(x0_3)
            return output

#Attention Unet
class Attention_Unet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision = False):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值
        self.relu = nn.ReLU(inplace=True)
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        self.AG1020 = ABD3377(nb_filter[1], nb_filter[2], nb_filter[2])
        self.Ag0010 = ABD3377(nb_filter[0], nb_filter[1], nb_filter[1])

        self.conv2_1 = VGGBlock(nb_filter[3]+nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        poolx1_0 = self.pool(x1_0)
        ag1020 = self.AG1020(pool = poolx1_0, x = x2_0)
        x2_1 = self.conv2_1(torch.cat([ag1020, self.up(x3_0)], 1))

        poolx0_0 = self.pool(x0_0)

        ag0010 = self.Ag0010(pool = poolx0_0, x = x1_0)
        x1_2 = self.conv1_2(torch.cat([ag0010, self.up(x2_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))
        output = self.final(x0_3)

        return output

#Attention-Unet++
class Attention_Unetjiajia(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #attention
        self.UpsampleSignal_1_0_0_0 = nn.Sequential(
                    nn.Conv2d(nb_filter[1], nb_filter[1], 1, padding = 0),
                    nn.BatchNorm2d(nb_filter[1])
        )
        self.EncoderFeature_0_0_1_0 = nn.Sequential(
                    nn.Conv2d(nb_filter[0], nb_filter[1], 1, padding = 0),
                    nn.BatchNorm2d(nb_filter[1])
        )
        self.channel_Process_1_0_0_0 = nn.Sequential(
                   nn.Conv2d(nb_filter[1], 1, 1, padding = 0),
                   nn.BatchNorm2d(1)
        )

        self.UpsampleSignal_2_0_1_0 = nn.Sequential(
            nn.Conv2d(nb_filter[2], nb_filter[2], 1, padding=0),
            nn.BatchNorm2d(nb_filter[2])
        )
        self.EncoderFeature_1_0_2_0 = nn.Sequential(
            nn.Conv2d(nb_filter[1], nb_filter[2], 1, padding=0),
            nn.BatchNorm2d(nb_filter[2])
        )
        self.channel_Process_2_0_1_0 = nn.Sequential(
            nn.Conv2d(nb_filter[2], 1, 1, padding=0),
            nn.BatchNorm2d(1)
        )

        self.UpsampleSignal_1_1_0_1 = nn.Sequential(
            nn.Conv2d(nb_filter[1], nb_filter[1], 1, padding=0),
            nn.BatchNorm2d(nb_filter[1])
        )
        self.EncoderFeature_0_1_1_1 = nn.Sequential(
            nn.Conv2d(nb_filter[0], nb_filter[1], 1, padding=0),
            nn.BatchNorm2d(nb_filter[1])
        )
        self.channel_Process_1_1_0_1 = nn.Sequential(
            nn.Conv2d(nb_filter[1], 1, 1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_11 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv1_1 = VGGBlock(nb_filter[2] + nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x_men = self.up(x1_0)
        x_men = self.UpsampleSignal_1_0_0_0(x_men)
        In_men = self.EncoderFeature_0_0_1_0(x0_0)
        x_cat = torch.add(x_men, In_men)
        x_cat = torch.relu(x_cat)
        x_jiangwei = self.channel_Process_1_0_0_0(x_cat)
        aerfa = torch.sigmoid(x_jiangwei)
        x1_00 = torch.mul(x0_0, aerfa)

        x0_1 = self.conv0_1(torch.cat([x1_00, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x_men2 = self.up(x2_0)
        x_men2 = self.UpsampleSignal_2_0_1_0(x_men2)
        In_men2 = self.EncoderFeature_1_0_2_0(x1_0)
        x_cat2 = torch.add(x_men2, In_men2)
        x_cat2 = torch.relu(x_cat2)
        x_jiangwei2 = self.channel_Process_2_0_1_0(x_cat2)
        aerfa2 = torch.sigmoid(x_jiangwei2)
        x2_10 = torch.mul(x1_0, aerfa2)
        x1_1 = self.conv1_1(torch.cat([x2_10, self.up(x2_0)], 1))

        x_men3 = self.up(x1_1)
        x_men3 = self.UpsampleSignal_1_1_0_1(x_men3)

        In_men3 = self.EncoderFeature_0_1_1_1(x0_1)

        x_cat3 = torch.add(x_men3, In_men3)
        x_cat3 = torch.relu(x_cat3)
        x_jiangwei3 = self.channel_Process_1_1_0_1(x_cat3)
        aerfa3 = torch.sigmoid(x_jiangwei3)
        x1_01 = torch.mul(x0_1, aerfa3)
        x0_2 = self.conv0_2(torch.cat([x1_01, x1_00, self.up(x1_1)], 1))
        output = self.final(x0_2)
        return output

#Sa-Unet
class SaAttention_Unet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision = False):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0_1 = encode_VGG(nb_filter[2], nb_filter[3])
        self.equal_att = new_equal(nb_filter[3], nb_filter[1] * nb_filter[1])
        self.conv3_0_2 = encode_VGG(nb_filter[3], nb_filter[3])

        self.conv2_1 = VGGBlock(nb_filter[3]+nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0_1 = self.conv3_0_1(self.pool(x2_0))
        att = self.equal_att(x3_0_1)
        x3_0_2 = self.conv3_0_2(att)
        #x3_0 = self.conv3_0(self.pool(x2_0))
        #x4_0 = self.conv4_0(self.pool(x3_0))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0_2)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))
        #x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_3)
        return output
class Attention_block2(nn.Module):
    def __init__(self, up_channel, x_channel, int_channel):
        super(Attention_block2, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(up_channel, int_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int_channel)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channel, int_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int_channel)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(int_channel, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, up, x):
        up1 = self.W_g(up)
        x1 = self.W_x(x)
        psi_ = self.relu(up1 + x1)
        psi = self.psi(psi_)

        return up*psi
class MaxDeepWise_Pool(torch.nn.MaxPool1d):
    def __init__(self, channels, isize):
        super(MaxDeepWise_Pool, self).__init__(channels)
        self.kernel_size = channels
        self.stride = isize

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = torch.nn.functional.max_pool1d(input, self.kernel_size, self.stride,
                                                self.padding, self.dilation, self.ceil_mode,
                                                self.return_indices)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)
class AvgDeepWise_Pool(torch.nn.AvgPool1d):
    def __init__(self, channels, isize):
        super(AvgDeepWise_Pool, self).__init__(channels)
        self.kernel_size = channels
        self.stride = isize

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = torch.nn.functional.avg_pool1d(input, self.kernel_size, self.stride,
                                                self.padding)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)

#FFnet
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class ASPP(nn.Module):
    def __init__(self, inplanes=1024, mid_c=256, dilations=[1, 6, 12, 18], factor=1):
        super(ASPP, self).__init__()
        # self.conv0 = nn.Conv2d(inplanes , inplanes // factor, 1, bias=False)
        self.aspp1 = _ASPPModule(inplanes // factor, mid_c, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes // factor, mid_c, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes // factor, mid_c, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes // factor, mid_c, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes // factor, mid_c, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(mid_c),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(mid_c * 5, inplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        # x = self.conv0(x)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out
class resconv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(resconv_block, self).__init__()
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)

        return x
class FFNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(FFNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = single_conv(ch_in=img_ch, ch_out=64)
        self.Conv2 = resconv_block(ch_in=64, ch_out=128)
        self.Conv3 = resconv_block(ch_in=128, ch_out=256)
        self.Conv4 = resconv_block(ch_in=256, ch_out=512)

        self.center = ASPP(1024, 256)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv1_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.Conv2_1x1 = nn.Conv2d(128, output_ch, kernel_size=1, stride=1, padding=0)
        self.Conv3_1x1 = nn.Conv2d(256, output_ch, kernel_size=1, stride=1, padding=0)
        self.Conv5_1x1 = nn.Conv2d(3 * output_ch, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x4 = self.center(x4)

        # decoding + concat path

        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        p1 = self.Conv1_1x1(d2)
        p2 = F.interpolate(self.Conv2_1x1(d3), size=p1.shape[2:], mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.Conv3_1x1(d4), size=p1.shape[2:], mode='bilinear', align_corners=False)

        p = torch.cat((p1, p2, p3), 1)
        p = self.Conv5_1x1(p)

        return p

#Method [23]
class Decode1AndGreen_transfer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
class Decode2(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, groups= groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
class Channel_attention_branch(nn.Module):
    def __init__(self, cat_channel):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.relu = nn.ReLU(inplace = True)
        self.m_v = nn.AdaptiveMaxPool2d(1)
        self.a_v = nn.AdaptiveAvgPool2d(1)
        self.attVector = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel // 2, kernel_size=1, stride = 1, padding = 0),
            nn.BatchNorm2d(cat_channel // 2)
        )
        self.attVector2 = nn.Sequential(
            nn.Conv2d(cat_channel // 2,  cat_channel // 2, kernel_size=1,  stride=1, padding=0),
            nn.BatchNorm2d(cat_channel // 2)
        )
    def forward(self, x):
        MaxPool1_1 = self.m_v(x)
        AVGPool1_1 = self.a_v(x)
        AVGVec = self.attVector(AVGPool1_1)
        MaxVec = self.attVector(MaxPool1_1)
        MaxPool1_1relu = self.relu(MaxVec)
        AVGPool1_1relu = self.relu(AVGVec)
        AVGVec2 = self.attVector2(AVGPool1_1relu)
        MaxVec2 = self.attVector2(MaxPool1_1relu)
        Wc = torch.sigmoid(MaxVec2 + AVGVec2)
        return Wc
class Spatial_attention_branch(nn.Module):
    def __init__(self, cat_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(cat_channel, 2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        Conv1 = self.conv1(x)
        Conv2 = self.conv2(self.relu(Conv1))
        Ws = torch.sigmoid(Conv2)
        return Ws
class AttentionBlockBiBM(nn.Module):
    def __init__(self, d1, d2):
        super().__init__()
        c  = d1 + d2
        self.W_s = Spatial_attention_branch(c)
        self.W_c = Channel_attention_branch(c)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, lowLevel, currentLevel):
        cat = torch.cat([lowLevel, currentLevel], 1)
        wc = self.W_c(cat)
        ws = self.W_s(cat)
        vecmul = torch.mul(wc, ws)
        currentmul = torch.mul(vecmul, currentLevel)
        add = torch.add(currentmul, lowLevel)
        return add
class AttentionBiBMNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #stage 1
        self.greenDecode_S1_0_0_1 = Decode1AndGreen_transfer(input_channels, nb_filter[0])
        self.Decode2_S1_0_0_2 = Decode2(nb_filter[0], nb_filter[0], 32)
        self.att_s1_0_0 = AttentionBlockBiBM(nb_filter[0], nb_filter[0])

        self.Decode2_S1_1_0_1 = Decode2(nb_filter[0], nb_filter[1], 32)
        self.Decode2_S1_1_0_2 = Decode2(nb_filter[1], nb_filter[1], 64)
        self.att_s1_1_0 = AttentionBlockBiBM(nb_filter[1], nb_filter[1])

        self.Decode2_S1_2_0_1 = Decode2(nb_filter[1], nb_filter[2], 64)
        self.Decode2_S1_2_0_2 = Decode2(nb_filter[2], nb_filter[2], 128)
        self.att_s1_2_0 = AttentionBlockBiBM(nb_filter[2], nb_filter[2])

        self.Decode2_S1_3_0_1 = Decode2(nb_filter[2], nb_filter[3], 128)
        self.Decode2_S1_3_0_2 = Decode2(nb_filter[3], nb_filter[3], 256)
        self.att_s1_3_0 = AttentionBlockBiBM(nb_filter[3], nb_filter[3])

        self.conv2_1 = nn.Conv2d(nb_filter[3], nb_filter[2], 1, padding=0)
        self.conv2_1_1 = Decode2(nb_filter[2] + nb_filter[2], nb_filter[2], 128)
        self.conv2_1_2 = Decode2(nb_filter[2], nb_filter[2], 128)

        self.conv1_1 = nn.Conv2d(nb_filter[2], nb_filter[1], 1, padding=0)
        self.conv1_1_1 = Decode2(nb_filter[1] + nb_filter[1], nb_filter[1], 64)
        self.conv1_1_2 = Decode2(nb_filter[1], nb_filter[1], 64)

        self.conv0_1 = nn.Conv2d(nb_filter[1], nb_filter[0], 1, padding=0)
        self.conv0_1_1 = Decode2(nb_filter[0] + nb_filter[0], nb_filter[0], 32)
        self.conv0_1_2 = Decode2(nb_filter[0], nb_filter[0], 32)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
    def forward(self, input):
        green0_0 = self.greenDecode_S1_0_0_1(input)
        decode0_0_1 = self.Decode2_S1_0_0_2(green0_0)
        att1 = self.att_s1_0_0(green0_0, decode0_0_1)

        pool_att1 = self.pool(att1)
        decode1_0_1 = self.Decode2_S1_1_0_1(pool_att1)
        decode1_0_2 = self.Decode2_S1_1_0_2(decode1_0_1)
        att2 = self.att_s1_1_0(decode1_0_1, decode1_0_2)

        pool_att2 = self.pool(att2)
        decode2_0_1 = self.Decode2_S1_2_0_1(pool_att2)
        decode2_0_2 = self.Decode2_S1_2_0_2(decode2_0_1)
        att3 = self.att_s1_2_0(decode2_0_1, decode2_0_2)

        pool_att3 = self.pool(att3)
        decode3_0_1 = self.Decode2_S1_3_0_1(pool_att3)
        decode3_0_2 = self.Decode2_S1_3_0_2(decode3_0_1)
        att4 = self.att_s1_3_0(decode3_0_1, decode3_0_2)

        up_att4 = self.up(att4)
        att21_11 = self.conv2_1(up_att4)
        decode2_1_1 = self.conv2_1_1(torch.cat([att21_11, att3], 1))
        decode2_1_2 = self.conv2_1_2(decode2_1_1)

        up_2_1_2 = self.up(decode2_1_2)
        att11_11 = self.conv1_1(up_2_1_2)
        decode1_1_1 = self.conv1_1_1(torch.cat([att11_11, att2], 1))
        decode1_1_2 = self.conv1_1_2(decode1_1_1)

        up_1_1_2 = self.up(decode1_1_2)
        att01_11 = self.conv0_1(up_1_1_2)
        decode0_1_1 = self.conv0_1_1(torch.cat([att01_11, att1], 1))
        decode0_1_2 = self.conv0_1_2(decode0_1_1)

        out = self.final(decode0_1_2)

        return out

#Our model : MDACM-Unet
class Aup1(nn.Module):
    def __init__(self, Maxpool, up, int_channel):
        super(Aup1, self).__init__()
        self.W_p1 = nn.Sequential(
            nn.Conv2d(Maxpool, int_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int_channel)
        )
        self.W_p2 = nn.Sequential(
            nn.Conv2d(up, int_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int_channel)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(int_channel, int_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int_channel)
        )
        self.psi1 = nn.Sequential(
            nn.Conv2d(int_channel, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.psi2 = nn.Sequential(
            nn.Conv2d(int_channel, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, pool, up, x):
        Pool = self.W_p1(pool)
        Up = self.W_p2(up)
        x1 = self.W_x(x)
        add1 = torch.add(Pool, x1)
        add2 = torch.add(Up, x1)
        act_1 = self.relu(add1)
        act_2 = self.relu(add2)
        psi_1 = self.psi1(act_1)
        psi_2 = self.psi2(act_2)
        cat = torch.cat([psi_1, psi_2], 1)
        psi_ = self.psi(cat)
        return x * psi_
class Aup2(nn.Module):
    def __init__(self, Maxpool, up, int_channel):
        super(Aup2, self).__init__()
        self.W_p1 = nn.Sequential(
            nn.Conv2d(Maxpool, int_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int_channel)
        )
        self.W_p2 = nn.Sequential(
            nn.Conv2d(up, int_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int_channel)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(int_channel, int_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int_channel)
        )
        self.psi1 = nn.Sequential(
            nn.Conv2d(int_channel, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.psi2 = nn.Sequential(
            nn.Conv2d(int_channel, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, pool, up, x):
        Pool = self.W_p1(pool)
        Up = self.W_p2(up)
        x1 = self.W_x(x)
        add1 = torch.add(Pool, x1)
        add2 = torch.add(Up, x1)
        act_1 = self.relu(add1)
        act_2 = self.relu(add2)
        psi_1 = self.psi1(act_1)
        psi_2 = self.psi2(act_2)
        cat = torch.cat([psi_1, psi_2], 1)
        psi_ = self.psi(cat)
        return x * psi_
class MDACM_Unet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision = False):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值
        self.relu = nn.ReLU(inplace=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        # self.drop1 = DropBlock2D(7, 0.9)
        # self.drop2 = DropBlock2D(5, 0.95)
        # self.drop3 = DropBlock2D(3, 1)
        self.drop_first = nn.Dropout(p=0.025)
        self.drop = nn.Dropout(p=0.05)
        self.AG1020 = Aup1(nb_filter[1], nb_filter[3], nb_filter[2])
        self.Ag0010 = Aup2(nb_filter[0], nb_filter[2], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[3]+nb_filter[2], nb_filter[2], nb_filter[2])

        self.conv1_2 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
    def forward(self, input):
        #测试
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.drop_first(self.pool(x0_0)))
        x2_0 = self.conv2_0(self.drop(self.pool(x1_0)))
        x3_0 = self.conv3_0(self.drop(self.pool(x2_0)))
        ag1020 = self.AG1020(self.pool(x1_0), self.up(x3_0), x2_0)
        x2_1 = self.conv2_1(self.drop(torch.cat([ag1020, self.up(x3_0)], 1)))
        ag0010 = self.Ag0010(self.pool(x0_0), self.up(x2_1), x1_0)
        x1_2 = self.conv1_2(self.drop(torch.cat([ag0010, self.up(x2_1)], 1)))
        x0_3 = self.conv0_3(self.drop(torch.cat([x0_0, self.up(x1_2)], 1)))
        out = self.final(x0_3)
        return out
        #训练
        # x0_0 = self.conv0_0(input)
        #
        # x1_0 = self.conv1_0(self.pool(x0_0))
        # x2_0 = self.conv2_0(self.pool(x1_0))
        # x3_0 = self.conv3_0(self.pool(x2_0))
        #
        # ag1020 = self.AG1020(self.pool(x1_0), self.up(x3_0), x2_0)
        # x2_1 = self.conv2_1(torch.cat([ag1020, self.up(self.drop1(x3_0))], 1))
        #
        # ag0010 = self.Ag0010(self.pool(x0_0), self.up(x2_1), x1_0)
        # x1_2 = self.conv1_2(torch.cat([ag0010, self.up(self.drop2(x2_1))], 1))
        #
        # x0_3 = self.conv0_3(torch.cat([x0_0, self.up(self.drop3(x1_2))], 1))
        #
        # out = self.final(x0_3)
        #
        # return out

#Ablation experiments
class ABD3377(nn.Module):
    def __init__(self, pool_channel, x_channel, int_channel):
        super(ABD3377, self).__init__()
        self.W_p = nn.Sequential(
            nn.Conv2d(pool_channel, int_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int_channel)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channel, int_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int_channel)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(int_channel, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pool, x):
        pool1 = self.W_p(pool)
        x1 = self.W_x(x)
        psi = self.relu(torch.add(pool1, x1))
        psi_ = self.psi(psi)
        return x * psi_
class new_equal(nn.Module):
    def __init__(self, _Channel, _H_W):
        super(new_equal, self).__init__()
        self.depthMax = MaxDeepWise_Pool(_Channel, _H_W)
        self.depthAvg = AvgDeepWise_Pool(_Channel, _H_W)
        self.psi = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input):
        max_pool = self.depthMax(input)
        avg_pool = self.depthAvg(input)
        cat_info = torch.cat([max_pool, avg_pool], 1)
        psi = self.psi(cat_info)
        return input * psi
class decode_VGG(nn.Module):
    def __init__(self, in_channels, middle_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out
class encode_VGG(nn.Module):
    def __init__(self, in_channels, middle_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        return out
#Ablation experiments_B
class Ablation_experiments_B(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision = False):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值
        self.relu = nn.ReLU(inplace=True)
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        self.AG1020 = ABD3377(nb_filter[1], nb_filter[2], nb_filter[2])
        self.Ag0010 = ABD3377(nb_filter[0], nb_filter[1], nb_filter[1])

        self.conv2_1 = VGGBlock(nb_filter[3]+nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        poolx1_0 = self.pool(x1_0)
        ag1020 = self.AG1020(pool = poolx1_0, x = x2_0)
        x2_1 = self.conv2_1(torch.cat([ag1020, self.up(x3_0)], 1))

        poolx0_0 = self.pool(x0_0)

        ag0010 = self.Ag0010(pool = poolx0_0, x = x1_0)
        x1_2 = self.conv1_2(torch.cat([ag0010, self.up(x2_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))
        output = self.final(x0_3)

        return output


#Ablation experiments_C
class Ablation_experiment_C(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision = False):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值
        self.relu = nn.ReLU(inplace=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0_1 = decode_VGG(nb_filter[2], nb_filter[3])
        self.equal_att = new_equal(nb_filter[3], nb_filter[1]*nb_filter[1])
        self.conv3_0_2 = encode_VGG(nb_filter[3], nb_filter[3])
        self.AG1020 = ABD3377(nb_filter[1], nb_filter[2], nb_filter[2])
        self.Ag0010 = ABD3377(nb_filter[0], nb_filter[1], nb_filter[1])

        self.conv2_1 = VGGBlock(nb_filter[3]+nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))

        x3_0_1 = self.conv3_0_1(self.pool(x2_0))
        att = self.equal_att(x3_0_1)
        x3_0_2 = self.conv3_0_2(att)

        ag1020 = self.AG1020(pool = self.pool(x1_0), x = x2_0)
        x2_1 = self.conv2_1(torch.cat([ag1020, self.up(x3_0_2)], 1))

        ag0010 = self.Ag0010(pool = self.pool(x0_0), x = x1_0)
        x1_2 = self.conv1_2(torch.cat([ag0010, self.up(x2_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))
        output = self.final(x0_3)

        return output