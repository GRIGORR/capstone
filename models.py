from config_utils import dict2config
from typing import List, Text

# from .SharedUtils import change_key
# from .cell_searchs import CellStructure, CellArchitectures


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
                              config.deconv, config.paper_arch)

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


# obtain the search space, i.e., a dict mapping the operation name into a python-function for this op
# def get_search_spaces(xtype, name) -> List[Text]:
#     if xtype == 'cell':
#         from cell_operations import SearchSpaceNames
#         assert name in SearchSpaceNames, 'invalid name [{:}] in {:}'.format(name, SearchSpaceNames.keys())
#         return SearchSpaceNames[name]
#     else:
#         raise ValueError('invalid search-space type is {:}'.format(xtype))