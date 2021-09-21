import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import functional as F

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adjs, afms):
        support = torch.bmm(adjs, afms)    # batch matrix product of adjacency matrix and atom feature matrix.
        result = torch.mm(support.view(-1, self.in_features), self.weight)
        output = result.view(-1, adjs.data.shape[1], self.out_features)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MolFP(Module):
    def __init__(self, out_features, bias=True):
        super(MolFP, self).__init__()
        self.out_features = out_features
        self.weight = Parameter(FloatTensor(1, out_features))
        if bias:
            self.bias = Parameter(FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.sum(input, 1)  # 0 is batchsize, 1 is atom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(1) + ' -> ' \
               + str(self.out_features) + ')'


class Dense(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class AFM_BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, bias=True):
        super(AFM_BatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine)
        self.weight = Parameter(FloatTensor(1, 1))
        if bias:
            self.bias = Parameter(FloatTensor(1))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn(x.contiguous())
        x = x.permute(0, 2, 1)
        return x


class Ave_multi_view(Module):
    def __init__(self, ave_source_num, feature_size, bias=False):
        super(Ave_multi_view, self).__init__()
        self.ave_source_num = ave_source_num
        self.feature_size = feature_size
        self.weight = Parameter(FloatTensor(ave_source_num))
        if bias:
            self.bias = Parameter(FloatTensor(feature_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        batch_size = input.data.shape[1]
        mol_size = input.data.shape[2]
        weight_expand = self.weight.view(self.ave_source_num, 1, 1, 1).expand(self.ave_source_num, batch_size, mol_size,
                                                                              self.feature_size)
        output = torch.sum(input * weight_expand, 0)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str('{} * {}'.format(self.ave_source_num, self.feature_size)) + ' -> ' \
               + str('1 * {}'.format(self.feature_size)) + ')'


class Atten(nn.Module):
    def __init__(self, hidden, in_feature, out_feature):
        super(Atten, self).__init__()
        self.w1 = nn.Parameter(FloatTensor(hidden, in_feature))
        self.w2 = nn.Parameter(FloatTensor(out_feature, hidden))
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.w1.size(0))
        stdv2 = 1. / math.sqrt(self.w2.size(0))
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)

    def forward(self, X):
        w1_expand = self.w1.unsqueeze(0)
        w2_expand = self.w2.unsqueeze(0)
        x = F.tanh(torch.matmul(w1_expand, X.transpose(1, 2)))
        x = torch.matmul(w2_expand, x)
        s = F.softmax(x)
        return torch.bmm(s, X)


class Concate_GCN(nn.Module):
    """
    @ The model used to train concatenate structure model
    @ Para:
    n_bfeat: num of types of 1st relation, atom pairs and atom self.
    n_afeat: length of atom features from RDkit
    n_sgc{i}_{j}: length of atom features updated by {j}th relation in {i}th layer
    n_den1 & n_den2: length of molecular feature updated by Dense Layer
    """

    def __init__(self, n_bfeat, n_afeat, n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                 n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5,
                 n_den1, n_den2, attention_bidden, out_feature,
                 nclass, dropout, use_att=True, molfp_mode='sum'):
        super(Concate_GCN, self).__init__()

        self.ngc1 = n_sgc1_1 + n_sgc1_2 + n_sgc1_3 + n_sgc1_4 + n_sgc1_5
        ngc1 = self.ngc1
        self.ngc2 = n_sgc2_1 + n_sgc2_2 + n_sgc2_3 + n_sgc2_4 + n_sgc2_5

        self.att1_1 = nn.Conv2d(n_bfeat, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_2 = nn.Conv2d(5, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_3 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_4 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_5 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

        self.soft = nn.Softmax(dim=2)

        self.att2_1 = nn.Conv2d(n_bfeat, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_2 = nn.Conv2d(5, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_3 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_4 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_5 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

        # Compress atom representations to mol representation
        if molfp_mode == 'sum':
            self.molfp1 = MolFP(self.ngc2)
            self.molfp2 = MolFP(self.ngc2)
            self.molfp = MolFP(self.ngc2)

        # Weighted average the mol representations to final mol representation.
        self.gc1_1 = GraphConvolution(n_afeat, n_sgc1_1, bias=True)
        self.gc1_2 = GraphConvolution(n_afeat, n_sgc1_2, bias=True)
        self.gc1_3 = GraphConvolution(n_afeat, n_sgc1_3, bias=True)
        self.gc1_4 = GraphConvolution(n_afeat, n_sgc1_4, bias=True)
        self.gc1_5 = GraphConvolution(n_afeat, n_sgc1_5, bias=True)

        self.gc2_1 = GraphConvolution(ngc1, n_sgc2_1, bias=True)
        self.gc2_2 = GraphConvolution(ngc1, n_sgc2_2, bias=True)
        self.gc2_3 = GraphConvolution(ngc1, n_sgc2_3, bias=True)
        self.gc2_4 = GraphConvolution(ngc1, n_sgc2_4, bias=True)
        self.gc2_5 = GraphConvolution(ngc1, n_sgc2_5, bias=True)

        self.den1 = Dense(self.ngc2, n_den1)
        self.den2 = Dense(n_den1, n_den2)
        self.den3 = Dense(n_den2, nclass)

        if use_cuda:
            self.att_bn1_1 = AFM_BatchNorm(n_sgc1_1).cuda()
            self.att_bn1_2 = AFM_BatchNorm(n_sgc1_2).cuda()
            self.att_bn1_3 = AFM_BatchNorm(n_sgc1_3).cuda()
            self.att_bn1_4 = AFM_BatchNorm(n_sgc1_4).cuda()
            self.att_bn1_5 = AFM_BatchNorm(n_sgc1_5).cuda()

            self.att_bn2_1 = AFM_BatchNorm(n_sgc2_1).cuda()
            self.att_bn2_2 = AFM_BatchNorm(n_sgc2_2).cuda()
            self.att_bn2_3 = AFM_BatchNorm(n_sgc2_3).cuda()
            self.att_bn2_4 = AFM_BatchNorm(n_sgc2_4).cuda()
            self.att_bn2_5 = AFM_BatchNorm(n_sgc2_5).cuda()

            self.molfp_bn = nn.BatchNorm1d(self.ngc2).cuda()
            self.bn_den1 = nn.BatchNorm1d(n_den1).cuda()
            self.bn_den2 = nn.BatchNorm1d(n_den2).cuda()
            self.att = Atten(attention_bidden, in_feature=self.ngc2, out_feature=out_feature).cuda()

        else:
            self.att_bn1_1 = AFM_BatchNorm(n_sgc1_1)
            self.att_bn1_2 = AFM_BatchNorm(n_sgc1_2)
            self.att_bn1_3 = AFM_BatchNorm(n_sgc1_3)
            self.att_bn1_4 = AFM_BatchNorm(n_sgc1_4)
            self.att_bn1_5 = AFM_BatchNorm(n_sgc1_5)

            self.att_bn2_1 = AFM_BatchNorm(n_sgc2_1)
            self.att_bn2_2 = AFM_BatchNorm(n_sgc2_2)
            self.att_bn2_3 = AFM_BatchNorm(n_sgc2_3)
            self.att_bn2_4 = AFM_BatchNorm(n_sgc2_4)
            self.att_bn2_5 = AFM_BatchNorm(n_sgc2_5)

            self.molfp_bn = nn.BatchNorm1d(self.ngc2)
            self.bn_den1 = nn.BatchNorm1d(n_den1)
            self.bn_den2 = nn.BatchNorm1d(n_den2)
            self.att = Atten(attention_bidden, in_feature=self.ngc2, out_feature=out_feature)

        self.dropout = dropout
        self.use_att = use_att

    def forward(self, adjs, afms, bfts, OrderAtt, AromAtt, ConjAtt, RingAtt):  # bfts
        mask = (1. - adjs) * -1e9
        mask_blank, _ = adjs.max(dim=2, keepdim=True)
        mask2 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], adjs.data.shape[2])
        mask3 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], self.ngc1)
        mask4 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], self.ngc2)

        A1_1 = self.att1_1(bfts.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_1 = self.soft(A1_1 + mask) * mask2
        x1_1 = self.gc1_1(A1_1, afms)
        x1_1 = F.relu(self.att_bn1_1(x1_1))
        x1_1 = F.dropout(x1_1, p=self.dropout, training=self.training)

        A1_2 = self.att1_2(OrderAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_2 = F.softmax(A1_2 + mask, 2) * mask2
        x1_2 = self.gc1_2(A1_2, afms)
        x1_2 = F.relu(self.att_bn1_2(x1_2))
        x1_2 = F.dropout(x1_2, p=self.dropout, training=self.training)

        A1_3 = self.att1_3(AromAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_3 = F.softmax(A1_3 + mask, 2) * mask2
        x1_3 = self.gc1_3(A1_3, afms)
        x1_3 = F.relu(self.att_bn1_3(x1_3))
        x1_3 = F.dropout(x1_3, p=self.dropout, training=self.training)

        A1_4 = self.att1_4(ConjAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_4 = F.softmax(A1_4 + mask, 2) * mask2
        x1_4 = self.gc1_4(A1_4, afms)
        x1_4 = F.relu(self.att_bn1_4(x1_4))
        x1_4 = F.dropout(x1_4, p=self.dropout, training=self.training)

        A1_5 = self.att1_5(RingAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_5 = F.softmax(A1_5 + mask, 2) * mask2
        x1_5 = self.gc1_5(A1_5, afms)
        x1_5 = F.relu(self.att_bn1_5(x1_5))
        x1_5 = F.dropout(x1_5, p=self.dropout, training=self.training)

        x1 = torch.cat((x1_1, x1_2, x1_3, x1_4, x1_5), dim=2) * mask3

        A2_1 = F.softmax(self.att2_1(bfts.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask, 2) * mask2
        x2_1 = self.gc2_1(A2_1, x1)
        x2_1 = F.relu(self.att_bn2_1(x2_1))
        x2_1 = F.dropout(x2_1, p=self.dropout, training=self.training)

        A2_2 = F.softmax(self.att2_2(OrderAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask,
                         2) * mask2
        x2_2 = self.gc2_2(A2_2, x1)
        x2_2 = F.relu(self.att_bn2_2(x2_2))
        x2_2 = F.dropout(x2_2, p=self.dropout, training=self.training)

        A2_3 = self.att2_3(AromAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A2_3 = F.softmax(A2_3 + mask, 2) * mask2
        x2_3 = self.gc2_3(A2_3, x1)
        x2_3 = F.relu(self.att_bn2_3(x2_3))
        x2_3 = F.dropout(x2_3, p=self.dropout, training=self.training)

        A2_4 = F.softmax(self.att2_4(ConjAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask,
                         2) * mask2
        x2_4 = self.gc2_4(A2_4, x1)
        x2_4 = F.relu(self.att_bn2_4(x2_4))
        x2_4 = F.dropout(x2_4, p=self.dropout, training=self.training)

        A2_5 = F.softmax(self.att2_5(RingAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask,
                         2) * mask2
        x2_5 = self.gc2_5(A2_5, x1)
        x2_5 = F.relu(self.att_bn2_5(x2_5))
        x2_5 = F.dropout(x2_5, p=self.dropout, training=self.training)

        x2 = torch.cat((x2_1, x2_2, x2_3, x2_4, x2_5), dim=2) * mask4  # (bs, n, 140)
        H = x2

        # x = self.molfp(x2)
        # x = self.molfp_bn(x)
        #
        # x = self.den1(x)
        # x = F.relu(self.bn_den1(x))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.den2(x)
        # x = F.relu(self.bn_den2(x))
        # x = self.den3(x)

        # x = self.att(H)

        x = self.molfp(x2)
        x = self.molfp_bn(x)

        x = self.den1(x)
        x = F.relu(self.bn_den1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.den2(x)
        x = F.relu(self.bn_den2(x))
        x = self.den3(x)

        return x


class Weighted_GCN(nn.Module):
    """
    @ The model used to train weighted sum structure model
    @ Para:
    n_bfeat: num of types of 1st relation, atom pairs and atom self.
    n_afeat: length of atom features from RDkit
    sum of n_sgc{i}_{j} over j: length of atom features updated by in {i}th layer for each relation.
    n_den1 & n_den2: length of molecular feature updated by Dense Layer
    """

    def __init__(self, n_bfeat, n_afeat, n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5,
                 n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5,
                 n_den1, n_den2,
                 nclass, dropout, use_att=True, molfp_mode='sum'):
        super(Weighted_GCN, self).__init__()

        self.ngc1 = n_sgc1_1 + n_sgc1_2 + n_sgc1_3 + n_sgc1_4 + n_sgc1_5
        ngc1 = self.ngc1
        self.ngc2 = n_sgc2_1 + n_sgc2_2 + n_sgc2_3 + n_sgc2_4 + n_sgc2_5
        ngc2 = self.ngc2

        self.att1_1 = nn.Conv2d(n_bfeat, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_2 = nn.Conv2d(5, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_3 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_4 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att1_5 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

        self.soft = nn.Softmax(dim=2)

        self.att2_1 = nn.Conv2d(n_bfeat, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_2 = nn.Conv2d(5, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_3 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_4 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.att2_5 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

        # Compress atom representations to mol representation
        self.molfp1 = MolFP(self.ngc2)  # 132 is the molsize after padding.
        self.molfp2 = MolFP(self.ngc2)
        self.molfp = MolFP(self.ngc2)

        # Weighted average the mol representations to final mol representation.
        self.gc1_1 = GraphConvolution(n_afeat, ngc1, bias=True)
        self.gc1_2 = GraphConvolution(n_afeat, ngc1, bias=True)
        self.gc1_3 = GraphConvolution(n_afeat, ngc1, bias=True)
        self.gc1_4 = GraphConvolution(n_afeat, ngc1, bias=True)
        self.gc1_5 = GraphConvolution(n_afeat, ngc1, bias=True)

        self.gc2_1 = GraphConvolution(ngc1, ngc2, bias=True)
        self.gc2_2 = GraphConvolution(ngc1, ngc2, bias=True)
        self.gc2_3 = GraphConvolution(ngc1, ngc2, bias=True)
        self.gc2_4 = GraphConvolution(ngc1, ngc2, bias=True)
        self.gc2_5 = GraphConvolution(ngc1, ngc2, bias=True)

        self.den1 = Dense(self.ngc2, n_den1)
        self.den2 = Dense(n_den1, n_den2)
        self.den3 = Dense(n_den2, nclass)

        if use_cuda:
            self.att_bn1_1 = AFM_BatchNorm(ngc1).cuda()
            self.att_bn1_2 = AFM_BatchNorm(ngc1).cuda()
            self.att_bn1_3 = AFM_BatchNorm(ngc1).cuda()
            self.att_bn1_4 = AFM_BatchNorm(ngc1).cuda()
            self.att_bn1_5 = AFM_BatchNorm(ngc1).cuda()

            self.att_bn2_1 = AFM_BatchNorm(ngc2).cuda()
            self.att_bn2_2 = AFM_BatchNorm(ngc2).cuda()
            self.att_bn2_3 = AFM_BatchNorm(ngc2).cuda()
            self.att_bn2_4 = AFM_BatchNorm(ngc2).cuda()
            self.att_bn2_5 = AFM_BatchNorm(ngc2).cuda()

            self.molfp_bn = nn.BatchNorm1d(self.ngc2).cuda()
            self.bn_den1 = nn.BatchNorm1d(n_den1).cuda()
            self.bn_den2 = nn.BatchNorm1d(n_den2).cuda()

        else:
            self.att_bn1_1 = AFM_BatchNorm(ngc1)
            self.att_bn1_2 = AFM_BatchNorm(ngc1)
            self.att_bn1_3 = AFM_BatchNorm(ngc1)
            self.att_bn1_4 = AFM_BatchNorm(ngc1)
            self.att_bn1_5 = AFM_BatchNorm(ngc1)

            self.att_bn2_1 = AFM_BatchNorm(ngc2)
            self.att_bn2_2 = AFM_BatchNorm(ngc2)
            self.att_bn2_3 = AFM_BatchNorm(ngc2)
            self.att_bn2_4 = AFM_BatchNorm(ngc2)
            self.att_bn2_5 = AFM_BatchNorm(ngc2)

            self.molfp_bn = nn.BatchNorm1d(self.ngc2)
            self.bn_den1 = nn.BatchNorm1d(n_den1)
            self.bn_den2 = nn.BatchNorm1d(n_den2)

        self.ave1 = Ave_multi_view(5, self.ngc1)
        self.ave2 = Ave_multi_view(5, self.ngc2)

        self.dropout = dropout
        self.use_att = use_att

    def forward(self, adjs, afms, bfts, OrderAtt, AromAtt, ConjAtt, RingAtt):  # bfts
        mask = (1. - adjs) * -1e9
        mask_blank, _ = adjs.max(dim=2, keepdim=True)
        mask2 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], adjs.data.shape[2])
        mask3 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], self.ngc1)
        mask4 = mask_blank.expand(adjs.data.shape[0], adjs.data.shape[1], self.ngc2)

        A1_1 = self.att1_1(bfts.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_1 = self.soft(A1_1 + mask) * mask2
        x1_1 = self.gc1_1(A1_1, afms)
        x1_1 = F.relu(self.att_bn1_1(x1_1))
        x1_1 = F.dropout(x1_1, p=self.dropout, training=self.training) * mask3

        A1_2 = self.att1_2(OrderAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_2 = F.softmax(A1_2 + mask, 2) * mask2
        x1_2 = self.gc1_2(A1_2, afms)
        x1_2 = F.relu(self.att_bn1_2(x1_2))
        x1_2 = F.dropout(x1_2, p=self.dropout, training=self.training) * mask3

        A1_3 = self.att1_3(AromAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_3 = F.softmax(A1_3 + mask, 2) * mask2
        x1_3 = self.gc1_3(A1_3, afms)
        x1_3 = F.relu(self.att_bn1_3(x1_3))
        x1_3 = F.dropout(x1_3, p=self.dropout, training=self.training) * mask3

        A1_4 = self.att1_4(ConjAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_4 = F.softmax(A1_4 + mask, 2) * mask2
        x1_4 = self.gc1_4(A1_4, afms)
        x1_4 = F.relu(self.att_bn1_4(x1_4))
        x1_4 = F.dropout(x1_4, p=self.dropout, training=self.training) * mask3

        A1_5 = self.att1_5(RingAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A1_5 = F.softmax(A1_5 + mask, 2) * mask2
        x1_5 = self.gc1_5(A1_5, afms)
        x1_5 = F.relu(self.att_bn1_5(x1_5))
        x1_5 = F.dropout(x1_5, p=self.dropout, training=self.training) * mask3

        x1 = torch.stack((x1_1, x1_2, x1_3, x1_4, x1_5), dim=0)
        x1 = self.ave1(x1)

        A2_1 = F.softmax(self.att2_1(bfts.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask, 2) * mask2
        x2_1 = self.gc2_1(A2_1, x1)
        x2_1 = F.relu(self.att_bn2_1(x2_1))
        x2_1 = F.dropout(x2_1, p=self.dropout, training=self.training) * mask4

        A2_2 = F.softmax(self.att2_2(OrderAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask,
                         2) * mask2
        x2_2 = self.gc2_2(A2_2, x1)
        x2_2 = F.relu(self.att_bn2_2(x2_2))
        x2_2 = F.dropout(x2_2, p=self.dropout, training=self.training) * mask4

        A2_3 = self.att2_3(AromAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1)
        A2_3 = F.softmax(A2_3 + mask, 2) * mask2
        x2_3 = self.gc2_3(A2_3, x1)
        x2_3 = F.relu(self.att_bn2_3(x2_3))
        x2_3 = F.dropout(x2_3, p=self.dropout, training=self.training) * mask4

        A2_4 = F.softmax(self.att2_4(ConjAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask,
                         2) * mask2
        x2_4 = self.gc2_4(A2_4, x1)
        x2_4 = F.relu(self.att_bn2_4(x2_4))
        x2_4 = F.dropout(x2_4, p=self.dropout, training=self.training) * mask4

        A2_5 = F.softmax(self.att2_5(RingAtt.float()).view(adjs.data.shape[0], adjs.data.shape[1], -1) + mask,
                         2) * mask2
        x2_5 = self.gc2_5(A2_5, x1)
        x2_5 = F.relu(self.att_bn2_5(x2_5))
        x2_5 = F.dropout(x2_5, p=self.dropout, training=self.training) * mask4

        x2 = torch.stack((x2_1, x2_2, x2_3, x2_4, x2_5), dim=0)
        x2 = self.ave2(x2)

        x = self.molfp(x2)
        x = self.molfp_bn(x)

        x = self.den1(x)
        x = F.relu(self.bn_den1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.den2(x)
        x = F.relu(self.bn_den2(x))
        x = self.den3(x)
        return x