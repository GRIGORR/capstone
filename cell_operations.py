import torch
import torch.nn as nn

OPS = {
    'none': lambda C_in, C_out, stride, affine, track_running_stats: Zero(C_in, C_out, stride),
    'avg_pool_3x3': lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, stride, 'avg', affine,
                                                                                     track_running_stats),
    'max_pool_3x3': lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, stride, 'max', affine,
                                                                                     track_running_stats),
    'nor_conv_7x7': lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (7, 7),
                                                                                        (stride, stride), (3, 3),
                                                                                        (1, 1), affine,
                                                                                        track_running_stats),
    'nor_conv_3x3': lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (3, 3),
                                                                                        (stride, stride), (1, 1),
                                                                                        (1, 1), affine,
                                                                                        track_running_stats),
    'nor_conv_1x1': lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (1, 1),
                                                                                        (stride, stride), (0, 0),
                                                                                        (1, 1), affine,
                                                                                        track_running_stats),
    'dua_sepc_3x3': lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(C_in, C_out, (3, 3),
                                                                                         (stride, stride), (1, 1),
                                                                                         (1, 1), affine,
                                                                                         track_running_stats),
    'dua_sepc_5x5': lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(C_in, C_out, (5, 5),
                                                                                         (stride, stride), (2, 2),
                                                                                         (1, 1), affine,
                                                                                         track_running_stats),
    'dil_sepc_3x3': lambda C_in, C_out, stride, affine, track_running_stats: SepConv(C_in, C_out, (3, 3),
                                                                                     (stride, stride), (2, 2), (2, 2),
                                                                                     affine, track_running_stats),
    'dil_sepc_5x5': lambda C_in, C_out, stride, affine, track_running_stats: SepConv(C_in, C_out, (5, 5),
                                                                                     (stride, stride), (4, 4), (2, 2),
                                                                                     affine, track_running_stats),
    'skip_connect': lambda C_in, C_out, stride, affine,
                           track_running_stats: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in,
                                                                                                                  C_out,
                                                                                                                  stride,
                                                                                                                  affine,
                                                                                                                  track_running_stats),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):
        return self.op(x)


class DualSepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
        super(DualSepConv, self).__init__()
        self.op_a = SepConv(C_in, C_in, kernel_size, stride, padding, dilation, affine, track_running_stats)
        self.op_b = SepConv(C_in, C_out, kernel_size, 1, padding, dilation, affine, track_running_stats)

    def forward(self, x):
        x = self.op_a(x)
        x = self.op_b(x)
        return x


class POOLING(nn.Module):

    def __init__(self, C_in, C_out, stride, mode, affine=True, track_running_stats=True):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine, track_running_stats)
        if mode == 'avg':
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == 'max':
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError('Invalid mode={:} in POOLING'.format(mode))

    def forward(self, inputs):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.)
            else:
                return x[:, :, ::self.stride, ::self.stride].mul(0.)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            # assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
        else:
            raise ValueError('Invalid stride : {:}'.format(stride))
        self.bn = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class GDAS_Reduction_Cell(nn.Module):

    def __init__(self, C_prev_prev, C_prev, C, reduction_prev, multiplier, affine, track_running_stats):
        super(GDAS_Reduction_Cell, self).__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, 2, affine, track_running_stats)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, 1, affine, track_running_stats)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, 1, affine, track_running_stats)
        self.multiplier = multiplier

        self.reduction = True
        self.ops1 = nn.ModuleList(
            [nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C, C, (1, 3), stride=(1, 2), padding=(0, 1), groups=8, bias=False),
                nn.Conv2d(C, C, (3, 1), stride=(2, 1), padding=(1, 0), groups=8, bias=False),
                nn.BatchNorm2d(C, affine=True),
                nn.ReLU(inplace=False),
                nn.Conv2d(C, C, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(C, affine=True)),
                nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C, C, (1, 3), stride=(1, 2), padding=(0, 1), groups=8, bias=False),
                    nn.Conv2d(C, C, (3, 1), stride=(2, 1), padding=(1, 0), groups=8, bias=False),
                    nn.BatchNorm2d(C, affine=True),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C, C, 1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(C, affine=True))])

        self.ops2 = nn.ModuleList(
            [nn.Sequential(
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.BatchNorm2d(C, affine=True)),
                nn.Sequential(
                    nn.MaxPool2d(3, stride=2, padding=1),
                    nn.BatchNorm2d(C, affine=True))])

    def forward_gdas(self, s0, s1, weightss=0, indexs=0):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        X0 = self.ops1[0](s0)
        X1 = self.ops1[1](s1)
        X2 = self.ops2[0](s0)
        X3 = self.ops2[1](s1)

        return torch.cat([X0, X1, X2, X3], dim=1)

