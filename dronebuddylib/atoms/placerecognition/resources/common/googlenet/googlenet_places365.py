import pkg_resources
import torch
import torch.nn as nn
import torch.nn.functional as F


class GoogLeNetPlaces365(nn.Module):

    def __init__(self, num_classes=6):
        super(GoogLeNetPlaces365, self).__init__()
        self.conv1_7x7_s2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), groups=1,
                                      bias=True)
        self.conv2_3x3_reduce = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                          bias=True)
        self.conv2_3x3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                   bias=True)
        self.inception_3a_1x1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                          bias=True)
        self.inception_3a_5x5_reduce = nn.Conv2d(in_channels=192, out_channels=16, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_3a_3x3_reduce = nn.Conv2d(in_channels=192, out_channels=96, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_3a_pool_proj = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=(1, 1), stride=(1, 1),
                                                groups=1, bias=True)
        self.inception_3a_5x5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), groups=1,
                                          bias=True)
        self.inception_3a_3x3 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                          bias=True)
        self.inception_3b_3x3_reduce = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_3b_1x1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1),
                                          groups=1, bias=True)
        self.inception_3b_5x5_reduce = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_3b_pool_proj = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1),
                                                groups=1, bias=True)
        self.inception_3b_3x3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(3, 3), stride=(1, 1),
                                          groups=1, bias=True)
        self.inception_3b_5x5 = nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(5, 5), stride=(1, 1), groups=1,
                                          bias=True)
        self.inception_4a_1x1 = nn.Conv2d(in_channels=480, out_channels=192, kernel_size=(1, 1), stride=(1, 1),
                                          groups=1, bias=True)
        self.inception_4a_3x3_reduce = nn.Conv2d(in_channels=480, out_channels=96, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_4a_5x5_reduce = nn.Conv2d(in_channels=480, out_channels=16, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_4a_pool_proj = nn.Conv2d(in_channels=480, out_channels=64, kernel_size=(1, 1), stride=(1, 1),
                                                groups=1, bias=True)
        self.inception_4a_3x3 = nn.Conv2d(in_channels=96, out_channels=208, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                          bias=True)
        self.inception_4a_5x5 = nn.Conv2d(in_channels=16, out_channels=48, kernel_size=(5, 5), stride=(1, 1), groups=1,
                                          bias=True)
        self.inception_4b_5x5_reduce = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_4b_1x1 = nn.Conv2d(in_channels=512, out_channels=160, kernel_size=(1, 1), stride=(1, 1),
                                          groups=1, bias=True)
        self.inception_4b_3x3_reduce = nn.Conv2d(in_channels=512, out_channels=112, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_4b_pool_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1),
                                                groups=1, bias=True)
        self.loss1_conv = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                    bias=True)
        self.inception_4b_5x5 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1,
                                          bias=True)
        self.inception_4b_3x3 = nn.Conv2d(in_channels=112, out_channels=224, kernel_size=(3, 3), stride=(1, 1),
                                          groups=1, bias=True)
        self.loss1_fc_1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.inception_4c_5x5_reduce = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_4c_1x1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1),
                                          groups=1, bias=True)
        self.inception_4c_3x3_reduce = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_4c_pool_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1),
                                                groups=1, bias=True)
        self.inception_4c_5x5 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1,
                                          bias=True)
        self.inception_4c_3x3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
                                          groups=1, bias=True)
        self.loss1_classifier_1 = nn.Linear(in_features=1024, out_features=365, bias=True)
        self.inception_4d_3x3_reduce = nn.Conv2d(in_channels=512, out_channels=144, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_4d_1x1 = nn.Conv2d(in_channels=512, out_channels=112, kernel_size=(1, 1), stride=(1, 1),
                                          groups=1, bias=True)
        self.inception_4d_5x5_reduce = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_4d_pool_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1),
                                                groups=1, bias=True)
        self.inception_4d_3x3 = nn.Conv2d(in_channels=144, out_channels=288, kernel_size=(3, 3), stride=(1, 1),
                                          groups=1, bias=True)
        self.inception_4d_5x5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1,
                                          bias=True)
        self.inception_4e_1x1 = nn.Conv2d(in_channels=528, out_channels=256, kernel_size=(1, 1), stride=(1, 1),
                                          groups=1, bias=True)
        self.inception_4e_5x5_reduce = nn.Conv2d(in_channels=528, out_channels=32, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_4e_3x3_reduce = nn.Conv2d(in_channels=528, out_channels=160, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.loss2_conv = nn.Conv2d(in_channels=528, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                    bias=True)
        self.inception_4e_pool_proj = nn.Conv2d(in_channels=528, out_channels=128, kernel_size=(1, 1), stride=(1, 1),
                                                groups=1, bias=True)
        self.inception_4e_5x5 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1,
                                          bias=True)
        self.inception_4e_3x3 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(3, 3), stride=(1, 1),
                                          groups=1, bias=True)
        self.loss2_fc_1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.inception_5a_1x1 = nn.Conv2d(in_channels=832, out_channels=256, kernel_size=(1, 1), stride=(1, 1),
                                          groups=1, bias=True)
        self.inception_5a_5x5_reduce = nn.Conv2d(in_channels=832, out_channels=32, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_5a_3x3_reduce = nn.Conv2d(in_channels=832, out_channels=160, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_5a_pool_proj = nn.Conv2d(in_channels=832, out_channels=128, kernel_size=(1, 1), stride=(1, 1),
                                                groups=1, bias=True)
        self.loss2_classifier_1 = nn.Linear(in_features=1024, out_features=365, bias=True)
        self.inception_5a_5x5 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1,
                                          bias=True)
        self.inception_5a_3x3 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(3, 3), stride=(1, 1),
                                          groups=1, bias=True)
        self.inception_5b_3x3_reduce = nn.Conv2d(in_channels=832, out_channels=192, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_5b_5x5_reduce = nn.Conv2d(in_channels=832, out_channels=48, kernel_size=(1, 1), stride=(1, 1),
                                                 groups=1, bias=True)
        self.inception_5b_1x1 = nn.Conv2d(in_channels=832, out_channels=384, kernel_size=(1, 1), stride=(1, 1),
                                          groups=1, bias=True)
        self.inception_5b_pool_proj = nn.Conv2d(in_channels=832, out_channels=128, kernel_size=(1, 1), stride=(1, 1),
                                                groups=1, bias=True)
        self.inception_5b_3x3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1),
                                          groups=1, bias=True)
        self.inception_5b_5x5 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1,
                                          bias=True)

        self.loss2_classifier_1 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)


    def forward(self, x):
        conv1_7x7_s2_pad = F.pad(x, (3, 3, 3, 3))
        conv1_7x7_s2 = self.conv1_7x7_s2(conv1_7x7_s2_pad)
        conv1_relu_7x7 = F.relu(conv1_7x7_s2)
        pool1_3x3_s2_pad = F.pad(conv1_relu_7x7, (0, 1, 0, 1), value=float('-inf'))
        pool1_3x3_s2 = F.max_pool2d(pool1_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        pool1_norm1 = F.local_response_norm(pool1_3x3_s2, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)
        conv2_3x3_reduce = self.conv2_3x3_reduce(pool1_norm1)
        conv2_relu_3x3_reduce = F.relu(conv2_3x3_reduce)
        conv2_3x3_pad = F.pad(conv2_relu_3x3_reduce, (1, 1, 1, 1))
        conv2_3x3 = self.conv2_3x3(conv2_3x3_pad)
        conv2_relu_3x3 = F.relu(conv2_3x3)
        conv2_norm2 = F.local_response_norm(conv2_relu_3x3, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)
        pool2_3x3_s2_pad = F.pad(conv2_norm2, (0, 1, 0, 1), value=float('-inf'))
        pool2_3x3_s2 = F.max_pool2d(pool2_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_3a_pool_pad = F.pad(pool2_3x3_s2, (1, 1, 1, 1), value=float('-inf'))
        inception_3a_pool = F.max_pool2d(inception_3a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                         ceil_mode=False)
        inception_3a_1x1 = self.inception_3a_1x1(pool2_3x3_s2)
        inception_3a_5x5_reduce = self.inception_3a_5x5_reduce(pool2_3x3_s2)
        inception_3a_3x3_reduce = self.inception_3a_3x3_reduce(pool2_3x3_s2)
        inception_3a_pool_proj = self.inception_3a_pool_proj(inception_3a_pool)
        inception_3a_relu_1x1 = F.relu(inception_3a_1x1)
        inception_3a_relu_5x5_reduce = F.relu(inception_3a_5x5_reduce)
        inception_3a_relu_3x3_reduce = F.relu(inception_3a_3x3_reduce)
        inception_3a_relu_pool_proj = F.relu(inception_3a_pool_proj)
        inception_3a_5x5_pad = F.pad(inception_3a_relu_5x5_reduce, (2, 2, 2, 2))
        inception_3a_5x5 = self.inception_3a_5x5(inception_3a_5x5_pad)
        inception_3a_3x3_pad = F.pad(inception_3a_relu_3x3_reduce, (1, 1, 1, 1))
        inception_3a_3x3 = self.inception_3a_3x3(inception_3a_3x3_pad)
        inception_3a_relu_5x5 = F.relu(inception_3a_5x5)
        inception_3a_relu_3x3 = F.relu(inception_3a_3x3)
        inception_3a_output = torch.cat(
            (inception_3a_relu_1x1, inception_3a_relu_3x3, inception_3a_relu_5x5, inception_3a_relu_pool_proj), 1)
        inception_3b_3x3_reduce = self.inception_3b_3x3_reduce(inception_3a_output)
        inception_3b_pool_pad = F.pad(inception_3a_output, (1, 1, 1, 1), value=float('-inf'))
        inception_3b_pool = F.max_pool2d(inception_3b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                         ceil_mode=False)
        inception_3b_1x1 = self.inception_3b_1x1(inception_3a_output)
        inception_3b_5x5_reduce = self.inception_3b_5x5_reduce(inception_3a_output)
        inception_3b_relu_3x3_reduce = F.relu(inception_3b_3x3_reduce)
        inception_3b_pool_proj = self.inception_3b_pool_proj(inception_3b_pool)
        inception_3b_relu_1x1 = F.relu(inception_3b_1x1)
        inception_3b_relu_5x5_reduce = F.relu(inception_3b_5x5_reduce)
        inception_3b_3x3_pad = F.pad(inception_3b_relu_3x3_reduce, (1, 1, 1, 1))
        inception_3b_3x3 = self.inception_3b_3x3(inception_3b_3x3_pad)
        inception_3b_relu_pool_proj = F.relu(inception_3b_pool_proj)
        inception_3b_5x5_pad = F.pad(inception_3b_relu_5x5_reduce, (2, 2, 2, 2))
        inception_3b_5x5 = self.inception_3b_5x5(inception_3b_5x5_pad)
        inception_3b_relu_3x3 = F.relu(inception_3b_3x3)
        inception_3b_relu_5x5 = F.relu(inception_3b_5x5)
        inception_3b_output = torch.cat(
            (inception_3b_relu_1x1, inception_3b_relu_3x3, inception_3b_relu_5x5, inception_3b_relu_pool_proj), 1)
        pool3_3x3_s2_pad = F.pad(inception_3b_output, (0, 1, 0, 1), value=float('-inf'))
        pool3_3x3_s2 = F.max_pool2d(pool3_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_4a_1x1 = self.inception_4a_1x1(pool3_3x3_s2)
        inception_4a_3x3_reduce = self.inception_4a_3x3_reduce(pool3_3x3_s2)
        inception_4a_5x5_reduce = self.inception_4a_5x5_reduce(pool3_3x3_s2)
        inception_4a_pool_pad = F.pad(pool3_3x3_s2, (1, 1, 1, 1), value=float('-inf'))
        inception_4a_pool = F.max_pool2d(inception_4a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                         ceil_mode=False)
        inception_4a_relu_1x1 = F.relu(inception_4a_1x1)
        inception_4a_relu_3x3_reduce = F.relu(inception_4a_3x3_reduce)
        inception_4a_relu_5x5_reduce = F.relu(inception_4a_5x5_reduce)
        inception_4a_pool_proj = self.inception_4a_pool_proj(inception_4a_pool)
        inception_4a_3x3_pad = F.pad(inception_4a_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4a_3x3 = self.inception_4a_3x3(inception_4a_3x3_pad)
        inception_4a_5x5_pad = F.pad(inception_4a_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4a_5x5 = self.inception_4a_5x5(inception_4a_5x5_pad)
        inception_4a_relu_pool_proj = F.relu(inception_4a_pool_proj)
        inception_4a_relu_3x3 = F.relu(inception_4a_3x3)
        inception_4a_relu_5x5 = F.relu(inception_4a_5x5)
        inception_4a_output = torch.cat(
            (inception_4a_relu_1x1, inception_4a_relu_3x3, inception_4a_relu_5x5, inception_4a_relu_pool_proj), 1)
        inception_4b_pool_pad = F.pad(inception_4a_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4b_pool = F.max_pool2d(inception_4b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                         ceil_mode=False)
        # loss1_ave_pool  = F.avg_pool2d(inception_4a_output, kernel_size=(5, 5), stride=(3, 3), padding=(0,), ceil_mode=True, count_include_pad=False)
        inception_4b_5x5_reduce = self.inception_4b_5x5_reduce(inception_4a_output)
        inception_4b_1x1 = self.inception_4b_1x1(inception_4a_output)
        inception_4b_3x3_reduce = self.inception_4b_3x3_reduce(inception_4a_output)
        inception_4b_pool_proj = self.inception_4b_pool_proj(inception_4b_pool)
        # loss1_conv      = self.loss1_conv(loss1_ave_pool)
        inception_4b_relu_5x5_reduce = F.relu(inception_4b_5x5_reduce)
        inception_4b_relu_1x1 = F.relu(inception_4b_1x1)
        inception_4b_relu_3x3_reduce = F.relu(inception_4b_3x3_reduce)
        inception_4b_relu_pool_proj = F.relu(inception_4b_pool_proj)
        # loss1_relu_conv = F.relu(loss1_conv)
        inception_4b_5x5_pad = F.pad(inception_4b_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4b_5x5 = self.inception_4b_5x5(inception_4b_5x5_pad)
        inception_4b_3x3_pad = F.pad(inception_4b_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4b_3x3 = self.inception_4b_3x3(inception_4b_3x3_pad)
        # loss1_fc_0      = loss1_relu_conv.view(loss1_relu_conv.size(0), -1)
        inception_4b_relu_5x5 = F.relu(inception_4b_5x5)
        inception_4b_relu_3x3 = F.relu(inception_4b_3x3)
        # loss1_fc_1      = self.loss1_fc_1(loss1_fc_0)
        inception_4b_output = torch.cat(
            (inception_4b_relu_1x1, inception_4b_relu_3x3, inception_4b_relu_5x5, inception_4b_relu_pool_proj), 1)
        # loss1_relu_fc   = F.relu(loss1_fc_1)
        inception_4c_5x5_reduce = self.inception_4c_5x5_reduce(inception_4b_output)
        inception_4c_pool_pad = F.pad(inception_4b_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4c_pool = F.max_pool2d(inception_4c_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                         ceil_mode=False)
        inception_4c_1x1 = self.inception_4c_1x1(inception_4b_output)
        inception_4c_3x3_reduce = self.inception_4c_3x3_reduce(inception_4b_output)
        # loss1_drop_fc   = F.dropout(input = loss1_relu_fc, p = 0.699999988079071, training = self.training, inplace = True)
        inception_4c_relu_5x5_reduce = F.relu(inception_4c_5x5_reduce)
        inception_4c_pool_proj = self.inception_4c_pool_proj(inception_4c_pool)
        inception_4c_relu_1x1 = F.relu(inception_4c_1x1)
        inception_4c_relu_3x3_reduce = F.relu(inception_4c_3x3_reduce)
        # loss1_classifier_0 = loss1_drop_fc.view(loss1_drop_fc.size(0), -1)
        inception_4c_5x5_pad = F.pad(inception_4c_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4c_5x5 = self.inception_4c_5x5(inception_4c_5x5_pad)
        inception_4c_relu_pool_proj = F.relu(inception_4c_pool_proj)
        inception_4c_3x3_pad = F.pad(inception_4c_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4c_3x3 = self.inception_4c_3x3(inception_4c_3x3_pad)
        # loss1_classifier_1 = self.loss1_classifier_1(loss1_classifier_0)
        inception_4c_relu_5x5 = F.relu(inception_4c_5x5)
        inception_4c_relu_3x3 = F.relu(inception_4c_3x3)
        inception_4c_output = torch.cat(
            (inception_4c_relu_1x1, inception_4c_relu_3x3, inception_4c_relu_5x5, inception_4c_relu_pool_proj), 1)
        inception_4d_pool_pad = F.pad(inception_4c_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4d_pool = F.max_pool2d(inception_4d_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                         ceil_mode=False)
        inception_4d_3x3_reduce = self.inception_4d_3x3_reduce(inception_4c_output)
        inception_4d_1x1 = self.inception_4d_1x1(inception_4c_output)
        inception_4d_5x5_reduce = self.inception_4d_5x5_reduce(inception_4c_output)
        inception_4d_pool_proj = self.inception_4d_pool_proj(inception_4d_pool)
        inception_4d_relu_3x3_reduce = F.relu(inception_4d_3x3_reduce)
        inception_4d_relu_1x1 = F.relu(inception_4d_1x1)
        inception_4d_relu_5x5_reduce = F.relu(inception_4d_5x5_reduce)
        inception_4d_relu_pool_proj = F.relu(inception_4d_pool_proj)
        inception_4d_3x3_pad = F.pad(inception_4d_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4d_3x3 = self.inception_4d_3x3(inception_4d_3x3_pad)
        inception_4d_5x5_pad = F.pad(inception_4d_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4d_5x5 = self.inception_4d_5x5(inception_4d_5x5_pad)
        inception_4d_relu_3x3 = F.relu(inception_4d_3x3)
        inception_4d_relu_5x5 = F.relu(inception_4d_5x5)
        inception_4d_output = torch.cat(
            (inception_4d_relu_1x1, inception_4d_relu_3x3, inception_4d_relu_5x5, inception_4d_relu_pool_proj), 1)
        inception_4e_1x1 = self.inception_4e_1x1(inception_4d_output)
        inception_4e_5x5_reduce = self.inception_4e_5x5_reduce(inception_4d_output)
        # loss2_ave_pool  = F.avg_pool2d(inception_4d_output, kernel_size=(5, 5), stride=(3, 3), padding=(0,), ceil_mode=True, count_include_pad=False)
        inception_4e_3x3_reduce = self.inception_4e_3x3_reduce(inception_4d_output)
        inception_4e_pool_pad = F.pad(inception_4d_output, (1, 1, 1, 1), value=float('-inf'))
        inception_4e_pool = F.max_pool2d(inception_4e_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                         ceil_mode=False)
        inception_4e_relu_1x1 = F.relu(inception_4e_1x1)
        inception_4e_relu_5x5_reduce = F.relu(inception_4e_5x5_reduce)
        # loss2_conv      = self.loss2_conv(loss2_ave_pool)
        inception_4e_relu_3x3_reduce = F.relu(inception_4e_3x3_reduce)
        inception_4e_pool_proj = self.inception_4e_pool_proj(inception_4e_pool)
        inception_4e_5x5_pad = F.pad(inception_4e_relu_5x5_reduce, (2, 2, 2, 2))
        inception_4e_5x5 = self.inception_4e_5x5(inception_4e_5x5_pad)
        # loss2_relu_conv = F.relu(loss2_conv)
        inception_4e_3x3_pad = F.pad(inception_4e_relu_3x3_reduce, (1, 1, 1, 1))
        inception_4e_3x3 = self.inception_4e_3x3(inception_4e_3x3_pad)
        inception_4e_relu_pool_proj = F.relu(inception_4e_pool_proj)
        inception_4e_relu_5x5 = F.relu(inception_4e_5x5)
        # loss2_fc_0      = loss2_relu_conv.view(loss2_relu_conv.size(0), -1)
        inception_4e_relu_3x3 = F.relu(inception_4e_3x3)
        # loss2_fc_1      = self.loss2_fc_1(loss2_fc_0)
        inception_4e_output = torch.cat(
            (inception_4e_relu_1x1, inception_4e_relu_3x3, inception_4e_relu_5x5, inception_4e_relu_pool_proj), 1)
        # loss2_relu_fc   = F.relu(loss2_fc_1)
        pool4_3x3_s2_pad = F.pad(inception_4e_output, (0, 1, 0, 1), value=float('-inf'))
        pool4_3x3_s2 = F.max_pool2d(pool4_3x3_s2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        # loss2_drop_fc   = F.dropout(input = loss2_relu_fc, p = 0.699999988079071, training = self.training, inplace = True)
        inception_5a_1x1 = self.inception_5a_1x1(pool4_3x3_s2)
        inception_5a_5x5_reduce = self.inception_5a_5x5_reduce(pool4_3x3_s2)
        inception_5a_pool_pad = F.pad(pool4_3x3_s2, (1, 1, 1, 1), value=float('-inf'))
        inception_5a_pool = F.max_pool2d(inception_5a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                         ceil_mode=False)
        inception_5a_3x3_reduce = self.inception_5a_3x3_reduce(pool4_3x3_s2)
        # loss2_classifier_0 = loss2_drop_fc.view(loss2_drop_fc.size(0), -1)
        inception_5a_relu_1x1 = F.relu(inception_5a_1x1)
        inception_5a_relu_5x5_reduce = F.relu(inception_5a_5x5_reduce)
        inception_5a_pool_proj = self.inception_5a_pool_proj(inception_5a_pool)
        inception_5a_relu_3x3_reduce = F.relu(inception_5a_3x3_reduce)
        # loss2_classifier_1 = self.loss2_classifier_1(loss2_classifier_0)
        inception_5a_5x5_pad = F.pad(inception_5a_relu_5x5_reduce, (2, 2, 2, 2))
        inception_5a_5x5 = self.inception_5a_5x5(inception_5a_5x5_pad)
        inception_5a_relu_pool_proj = F.relu(inception_5a_pool_proj)
        inception_5a_3x3_pad = F.pad(inception_5a_relu_3x3_reduce, (1, 1, 1, 1))
        inception_5a_3x3 = self.inception_5a_3x3(inception_5a_3x3_pad)
        inception_5a_relu_5x5 = F.relu(inception_5a_5x5)
        inception_5a_relu_3x3 = F.relu(inception_5a_3x3)
        inception_5a_output = torch.cat(
            (inception_5a_relu_1x1, inception_5a_relu_3x3, inception_5a_relu_5x5, inception_5a_relu_pool_proj), 1)
        inception_5b_3x3_reduce = self.inception_5b_3x3_reduce(inception_5a_output)
        inception_5b_pool_pad = F.pad(inception_5a_output, (1, 1, 1, 1), value=float('-inf'))
        inception_5b_pool = F.max_pool2d(inception_5b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                         ceil_mode=False)
        inception_5b_5x5_reduce = self.inception_5b_5x5_reduce(inception_5a_output)
        inception_5b_1x1 = self.inception_5b_1x1(inception_5a_output)
        inception_5b_relu_3x3_reduce = F.relu(inception_5b_3x3_reduce)
        inception_5b_pool_proj = self.inception_5b_pool_proj(inception_5b_pool)
        inception_5b_relu_5x5_reduce = F.relu(inception_5b_5x5_reduce)
        inception_5b_relu_1x1 = F.relu(inception_5b_1x1)
        inception_5b_3x3_pad = F.pad(inception_5b_relu_3x3_reduce, (1, 1, 1, 1))
        inception_5b_3x3 = self.inception_5b_3x3(inception_5b_3x3_pad)
        inception_5b_relu_pool_proj = F.relu(inception_5b_pool_proj)
        inception_5b_5x5_pad = F.pad(inception_5b_relu_5x5_reduce, (2, 2, 2, 2))
        inception_5b_5x5 = self.inception_5b_5x5(inception_5b_5x5_pad)
        inception_5b_relu_3x3 = F.relu(inception_5b_3x3)
        inception_5b_relu_5x5 = F.relu(inception_5b_5x5)
        inception_5b_output = torch.cat(
            (inception_5b_relu_1x1, inception_5b_relu_3x3, inception_5b_relu_5x5, inception_5b_relu_pool_proj), 1)
        pool5_7x7_s1 = F.avg_pool2d(inception_5b_output, kernel_size=(7, 7), stride=(1, 1), padding=(0,),
                                    ceil_mode=False, count_include_pad=False)
        pool5_drop_7x7_s1 = F.dropout(input=pool5_7x7_s1, p=0.4000000059604645, training=self.training, inplace=True)
        return pool5_drop_7x7_s1  # , loss2_classifier_1, loss1_classifier_1


cnn = GoogLeNetPlaces365()
model_path = pkg_resources.resource_filename(__name__, "/googlenet_places365.pth")
cnn.load_state_dict(torch.load(model_path, map_location='cpu'))
cnn.eval()

