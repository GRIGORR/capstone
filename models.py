from config_utils import dict2config
from typing import List, Text

# from .SharedUtils import change_key
# from .cell_searchs import CellStructure, CellArchitectures

import torch
import torch.nn as nn
from cell_operations import OPS
import os


# Cell-based NAS Models
def get_cell_based_tiny_net(config):
    # if isinstance(config, dict):
    #     config = dict2config(config, None)  # to support the argument being a dict
    super_type = config.super_type  # getattr(config, 'super_type', 'basic')
    group_names = ['DARTS-V1', 'DARTS-V2', 'GDAS', 'SETN', 'ENAS', 'RANDOM']
    # if super_type == 'basic' and config.name in group_names:
    #     from .cell_searchs import nas201_super_nets as nas_super_nets
    #     try:
    #         return nas_super_nets[config.name](config.C, config.N, config.max_nodes, config.num_classes, config.space,
    #                                            config.affine, config.track_running_stats)
    #     except:
    #         return nas_super_nets[config.name](config.C, config.N, config.max_nodes, config.num_classes, config.space)
    # elif super_type == 'nasnet-super':
    if super_type == 'nasnet-super':
        # from .cell_searchs import nasnet_super_nets as nas_super_nets
        # return nas_super_nets[config.name](config.C, config.N, config.steps, config.multiplier, \
        #                                    config.stem_multiplier, config.num_classes, config.space, config.affine, config.track_running_stats)

        from search_model_gdas_nasnet import NASNetworkGDAS
        return NASNetworkGDAS(config.channel, config.num_cells, config.steps, config.multiplier,
                              config.stem_multiplier, config.num_classes, config.space,
                              config.affine, config.track_running_stats, config.fix_reduction,
                              config.deconv, config.paper_arch, config.no_gumbel)

    # elif config.name == 'infer.tiny':
    #     from .cell_infers import TinyNetwork
    #     if hasattr(config, 'genotype'):
    #         genotype = config.genotype
    #     elif hasattr(config, 'arch_str'):
    #         genotype = CellStructure.str2structure(config.arch_str)
    #     else: raise ValueError('Can not find genotype from this config : {:}'.format(config))
    #     return TinyNetwork(config.C, config.N, genotype, config.num_classes)
    # elif config.name == 'infer.shape.tiny':
    #     from .shape_infers import DynamicShapeTinyNet
    #     if isinstance(config.channels, str):
    #         channels = tuple([int(x) for x in config.channels.split(':')])
    #     else: channels = config.channels
    #     genotype = CellStructure.str2structure(config.genotype)
    #     return DynamicShapeTinyNet(channels, genotype, config.num_classes)
    # elif config.name == 'infer.nasnet-cifar':
    #     from .cell_infers import NASNetonCIFAR
    #     raise NotImplementedError
    else:
        raise ValueError('invalid network name : {:}'.format(config.name))


# Try to obtain the network by config.
# def obtain_model(config, extra_path=None):
#     if config.dataset == 'cifar':
#         return get_cifar_models(config, extra_path)
#     elif config.dataset == 'imagenet':
#         return get_imagenet_models(config)
#     else:
#         raise ValueError('invalid dataset in the model config : {:}'.format(config))


def get_cifar_models(config):
    # super_type = getattr(config, 'super_type', 'basic')
    model_path = config.model_path
    # if super_type == 'basic':
    #     from .CifarResNet      import CifarResNet
    #     from .CifarDenseNet    import DenseNet
    #     from .CifarWideResNet  import CifarWideResNet
    #     if config.arch == 'resnet':
    #         return CifarResNet(config.module, config.depth, config.class_num, config.zero_init_residual)
    #     elif config.arch == 'densenet':
    #         return DenseNet(config.growthRate, config.depth, config.reduction, config.class_num, config.bottleneck)
    #     elif config.arch == 'wideresnet':
    #         return CifarWideResNet(config.depth, config.wide_factor, config.class_num, config.dropout)
    #     else:
    #         raise ValueError('invalid module type : {:}'.format(config.arch))
    # elif super_type.startswith('infer'):
    # from .shape_infers import InferWidthCifarResNet
    # from .shape_infers import InferDepthCifarResNet
    # from .shape_infers import InferCifarResNet
    # assert len(super_type.split('-')) == 2, 'invalid super_type : {:}'.format(super_type)
    # infer_mode = super_type.split('-')[1]
    # if infer_mode == 'width':
    #     return InferWidthCifarResNet(config.module, config.depth, config.xchannels, config.class_num, config.zero_init_residual)
    # elif infer_mode == 'depth':
    #     return InferDepthCifarResNet(config.module, config.depth, config.xblocks, config.class_num, config.zero_init_residual)
    # elif infer_mode == 'shape':
    #     return InferCifarResNet(config.module, config.depth, config.xblocks, config.xchannels, config.class_num, config.zero_init_residual)
    # elif infer_mode == 'nasnet.cifar':
    # genotype = config.genotype
    # if extra_path is not None:  # reload genotype by extra_path

    xdata = torch.load(model_path)
    current_epoch = xdata['epoch']
    genotype = xdata['genotypes'][current_epoch - 1]
    return NASNetonCIFAR(config.ichannel, config.layers, config.stem_multi, config.class_num, genotype,
                         config.auxiliary)
    # else:
    #     raise ValueError('invalid infer-mode : {:}'.format(infer_mode))
    # else:
    #     raise ValueError('invalid super-type : {:}'.format(super_type))


# obtain the search space, i.e., a dict mapping the operation name into a python-function for this op
# def get_search_spaces(xtype, name) -> List[Text]:
#     if xtype == 'cell':
#         from cell_operations import SearchSpaceNames
#         assert name in SearchSpaceNames, 'invalid name [{:}] in {:}'.format(name, SearchSpaceNames.keys())
#         return SearchSpaceNames[name]
#     else:
#         raise ValueError('invalid search-space type is {:}'.format(xtype))


# The macro structure is based on NASNet
class NASNetonCIFAR(nn.Module):

    def __init__(self, C, N, stem_multiplier, num_classes, genotype, auxiliary, affine=True, track_running_stats=True):
        super(NASNetonCIFAR, self).__init__()
        self._C = C
        self._layerN = N
        self.stem = nn.Sequential(
            nn.Conv2d(3, C * stem_multiplier, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C * stem_multiplier))

        # config for each layer
        layer_channels = [C] * N + [C * 2] + [C * 2] * (N - 1) + [C * 4] + [C * 4] * (N - 1)
        layer_reductions = [False] * N + [True] + [False] * (N - 1) + [True] + [False] * (N - 1)

        C_prev_prev, C_prev, C_curr, reduction_prev = C * stem_multiplier, C * stem_multiplier, C, False
        self.auxiliary_index = None
        self.auxiliary_head = None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            cell = NASNetInferCell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, affine,
                                   track_running_stats)
            self.cells.append(cell)
            C_prev_prev, C_prev, reduction_prev = C_prev, cell._multiplier * C_curr, reduction
            if reduction and C_curr == C * 4 and auxiliary:
                self.auxiliary_head = AuxiliaryHeadCIFAR(C_prev, num_classes)
                self.auxiliary_index = index
        self._Layer = len(self.cells)
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.drop_path_prob = -1

    def update_drop_path(self, drop_path_prob):
        self.drop_path_prob = drop_path_prob

    def auxiliary_param(self):
        if self.auxiliary_head is None:
            return []
        else:
            return list(self.auxiliary_head.parameters())

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return ('{name}(C={_C}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

    def forward(self, inputs):
        stem_feature, logits_aux = self.stem(inputs), None
        cell_results = [stem_feature, stem_feature]
        for i, cell in enumerate(self.cells):
            cell_feature = cell(cell_results[-2], cell_results[-1], self.drop_path_prob)
            cell_results.append(cell_feature)
            if self.auxiliary_index is not None and i == self.auxiliary_index and self.training:
                logits_aux = self.auxiliary_head(cell_results[-1])
        out = self.lastact(cell_results[-1])
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        if logits_aux is None:
            return out, logits
        else:
            return out, [logits, logits_aux]


class NASNetInferCell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, affine, track_running_stats):
        super(NASNetInferCell, self).__init__()
        self.reduction = reduction
        if reduction_prev:
            self.preprocess0 = OPS['skip_connect'](C_prev_prev, C, 2, affine, track_running_stats)
        else:
            self.preprocess0 = OPS['nor_conv_1x1'](C_prev_prev, C, 1, affine, track_running_stats)
        self.preprocess1 = OPS['nor_conv_1x1'](C_prev, C, 1, affine, track_running_stats)

        if not reduction:
            nodes, concats = genotype['normal'], genotype['normal_concat']
        else:
            nodes, concats = genotype['reduce'], genotype['reduce_concat']
        self._multiplier = len(concats)
        self._concats = concats
        self._steps = len(nodes)
        self._nodes = nodes
        self.edges = nn.ModuleDict()
        for i, node in enumerate(nodes):
            for in_node in node:
                name, j = in_node[0], in_node[1]
                stride = 2 if reduction and j < 2 else 1
                node_str = '{:}<-{:}'.format(i + 2, j)
                self.edges[node_str] = OPS[name](C, C, stride, affine, track_running_stats)

    # [TODO] to support drop_prob in this function..
    def forward(self, s0, s1, unused_drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i, node in enumerate(self._nodes):
            clist = []
            for in_node in node:
                name, j = in_node[0], in_node[1]
                node_str = '{:}<-{:}'.format(i + 2, j)
                op = self.edges[node_str]
                clist.append(op(states[j]))
            states.append(sum(clist))
        return torch.cat([states[x] for x in self._concats], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x
