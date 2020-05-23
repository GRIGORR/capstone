import os, json
from os import path as osp
from pathlib import Path
from collections import namedtuple
import argparse
import random

support_types = ('str', 'int', 'bool', 'float', 'none')

import os, sys, time, random, argparse

def add_shared_args( parser ):
  # Data Generation
  parser.add_argument('--dataset',          type=str,                   help='The dataset name.')
  parser.add_argument('--data_path',        type=str,                   help='The dataset name.')
  parser.add_argument('--cutout_length',    type=int,                   help='The cutout length, negative means not use.')
  # Printing
  parser.add_argument('--print_freq',       type=int,   default=100,    help='print frequency (default: 200)')
  parser.add_argument('--print_freq_eval',  type=int,   default=100,    help='print frequency (default: 200)')
  # Checkpoints
  parser.add_argument('--eval_frequency',   type=int,   default=1,      help='evaluation frequency (default: 200)')
  parser.add_argument('--save_dir',         type=str,                   help='Folder to save checkpoints and log.')
  # Acceleration
  parser.add_argument('--workers',          type=int,   default=8,      help='number of data loading workers (default: 8)')
  # Random Seed
  parser.add_argument('--rand_seed',        type=int,   default=-1,     help='manual seed')



def obtain_basic_args():
  parser = argparse.ArgumentParser(description='Train a classification model on typical image classification datasets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--resume'      ,     type=str,                   help='Resume path.')
  parser.add_argument('--init_model'  ,     type=str,                   help='The initialization model path.')
  parser.add_argument('--model_config',     type=str,                   help='The path to the model configuration')
  parser.add_argument('--optim_config',     type=str,                   help='The path to the optimizer configuration')
  parser.add_argument('--procedure'   ,     type=str,                   help='The procedure basic prefix.')
  parser.add_argument('--model_source',     type=str,  default='normal',help='The source of model defination.')
  parser.add_argument('--extra_model_path', type=str,  default=None,    help='The extra model ckp file (help to indicate the searched architecture).')
  add_shared_args( parser )
  # Optimization options
  parser.add_argument('--batch_size',       type=int,  default=2,       help='Batch size for training.')
  args = parser.parse_args()

  if args.rand_seed is None or args.rand_seed < 0:
    args.rand_seed = random.randint(1, 100000)
  assert args.save_dir is not None, 'save-path argument can not be None'
  return args



def convert_param(original_lists):
    assert isinstance(original_lists, list), 'The type is not right : {:}'.format(original_lists)
    ctype, value = original_lists[0], original_lists[1]
    assert ctype in support_types, 'Ctype={:}, support={:}'.format(ctype, support_types)
    is_list = isinstance(value, list)
    if not is_list: value = [value]
    outs = []
    for x in value:
        if ctype == 'int':
            x = int(x)
        elif ctype == 'str':
            x = str(x)
        elif ctype == 'bool':
            x = bool(int(x))
        elif ctype == 'float':
            x = float(x)
        elif ctype == 'none':
            if x.lower() != 'none':
                raise ValueError('For the none type, the value must be none instead of {:}'.format(x))
            x = None
        else:
            raise TypeError('Does not know this type : {:}'.format(ctype))
        outs.append(x)
    if not is_list: outs = outs[0]
    return outs


def load_config(path, extra, logger):
    # path = str(path)
    if hasattr(logger, 'log'): logger.log(path)
    assert os.path.exists(path), 'Can not find {:}'.format(path)
    # Reading data back
    with open(path, 'r') as f:
        data = json.load(f)
    content = {k: convert_param(v) for k, v in data.items()}
    assert extra is None or isinstance(extra, dict), 'invalid type of extra : {:}'.format(extra)
    if isinstance(extra, dict):
        content = {**content, **extra}
    Arguments = namedtuple('Configure', ' '.join(content.keys()))
    content = Arguments(**content)
    if hasattr(logger, 'log'): logger.log('{:}'.format(content))
    return content


def configure2str(config, xpath=None):
    if not isinstance(config, dict):
        config = config.asdict()

    def cstring(x):
        return "\"{:}\"".format(x)

    def gtype(x):
        if isinstance(x, list): x = x[0]
        if isinstance(x, str):
            return 'str'
        elif isinstance(x, bool):
            return 'bool'
        elif isinstance(x, int):
            return 'int'
        elif isinstance(x, float):
            return 'float'
        elif x is None:
            return 'none'
        else:
            raise ValueError('invalid : {:}'.format(x))

    def cvalue(x, xtype):
        if isinstance(x, list):
            is_list = True
        else:
            is_list, x = False, [x]
        temps = []
        for temp in x:
            if xtype == 'bool':
                temp = cstring(int(temp))
            elif xtype == 'none':
                temp = cstring('None')
            else:
                temp = cstring(temp)
            temps.append(temp)
        if is_list:
            return "[{:}]".format(', '.join(temps))
        else:
            return temps[0]

    xstrings = []
    for key, value in config.items():
        xtype = gtype(value)
        string = '  {:20s} : [{:8s}, {:}]'.format(cstring(key), cstring(xtype), cvalue(value, xtype))
        xstrings.append(string)
    Fstring = '{\n' + ',\n'.join(xstrings) + '\n}'
    if xpath is not None:
        parent = Path(xpath).resolve().parent
        parent.mkdir(parents=True, exist_ok=True)
        if osp.isfile(xpath): os.remove(xpath)
        with open(xpath, "w") as text_file:
            text_file.write('{:}'.format(Fstring))
    return Fstring


def dict2config(xdict, logger):
    assert isinstance(xdict, dict), 'invalid type : {:}'.format(type(xdict))
    Arguments = namedtuple('Configure', ' '.join(xdict.keys()))
    content = Arguments(**xdict)
    if hasattr(logger, 'log'): logger.log('{:}'.format(content))
    return content
