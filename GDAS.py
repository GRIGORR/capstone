import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# from apex import amp
import json
import argparse
from copy import deepcopy
import numpy as np
from config_utils import load_config, dict2config
from get_dataset_with_transform import get_datasets, get_nas_search_loaders
from procedures import prepare_logger, save_checkpoint, copy_checkpoint
from utils import obtain_accuracy
from tqdm import tqdm
from models import get_cell_based_tiny_net
from torch.utils.tensorboard import SummaryWriter


def search_func(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, writer, epoch, mixed_prec=False):
    base_losses, base_top1, = 0, 0
    arch_losses, arch_top1, = 0, 0
    network.train()
    train_data = len(xloader)
    for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
        base_targets = base_targets.cuda(non_blocking=True)
        arch_targets = arch_targets.cuda(non_blocking=True)
        base_inputs = base_inputs.cuda(non_blocking=True)
        arch_inputs = arch_inputs.cuda(non_blocking=True)
        
        # update the weights
        scheduler.step()
        w_optimizer.zero_grad()
        _, logits = network(base_inputs)
        base_loss = criterion(logits, base_targets)
        if mixed_prec:
            with amp.scale_loss(base_loss, w_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            base_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        w_optimizer.step()
        # record
        base_prec1 = obtain_accuracy(logits.data, base_targets.data, topk=(1,))
        base_losses += base_loss.item()
        base_top1 += base_prec1[0].item()

        # update the architecture-weight
        a_optimizer.zero_grad()
        _, logits = network(arch_inputs)
        arch_loss = criterion(logits, arch_targets)
        if mixed_prec:
            with amp.scale_loss(arch_loss, a_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            arch_loss.backward()
        a_optimizer.step()
        # record
        arch_prec1 = obtain_accuracy(logits.data, arch_targets.data, topk=(1,))
        arch_losses += arch_loss.item()
        arch_top1 += arch_prec1[0].item()

        writer.add_scalars('Learning_rate', {'lr_w': w_optimizer.state_dict()['param_groups'][0]['lr']},
                           epoch * train_data + step)
        writer.add_scalars('Learning_rate', {'lr_a': a_optimizer.state_dict()['param_groups'][0]['lr']},
                           epoch * train_data + step)
        writer.add_scalars('Loss', {'Weight_loss': base_loss.item()}, epoch * train_data + step)
        writer.add_scalars('Loss', {'Arch_loss': arch_loss.item()}, epoch * train_data + step)
        writer.add_scalars('Loss', {'Mean Loss': (arch_loss.item() + base_loss.item())/2}, epoch * train_data + step)

        writer.add_scalars('Accuracy_1', {'Weight_acc': base_prec1[0].item()}, epoch * train_data + step)
        writer.add_scalars('Accuracy_1', {'Arch_acc': arch_prec1[0].item()}, epoch * train_data + step)
        writer.add_scalars('Accuracy_1', {'Mean Accuracy': (arch_prec1[0].item() + base_prec1[0].item())/2}, epoch * train_data + step)

    return arch_top1 / train_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser("GDAS")
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    # channels and number-of-cells
    parser.add_argument('--max_nodes', type=int, help='The maximum number of nodes.') # ???????
    parser.add_argument('--channel', type=int, default=16,  help='The number of channels.')
    parser.add_argument('--num_cells', type=int, default=2, help='The number of cells in one block.')
    parser.add_argument('--steps', type=int, default=4, help='The number of nodes in one cell.')
    parser.add_argument('--multiplier', type=int, default=4, help='The number of nodes to concat in the end of a cell.')
    parser.add_argument('--stem_multiplier', type=int, default=3,
                        help='Input will be converted from 3 channels to stem_multiplier*channels')
    parser.add_argument('--track_running_stats', type=bool, default=True, choices=[True, False],
                        help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--affine', type=bool, default=False, choices=[True, False],
                        help='affine or not in the BN layer.') ## delete?
    parser.add_argument('--super_type', type=str, default='nasnet-super',
                        help='Macro structure of the network.')  ## may be deletable
    parser.add_argument('--LR', type=float, default=0.025, help='learning rate for weights in layers')
    parser.add_argument('--decay', type=float, default=0.0005, help='decay rate for weights in layers')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--nesterov', type=bool, default=True, choices=[True, False], help='nesterov in SGD')
    parser.add_argument('--eta_min', type=float, default=0.001, help='Min value for eta')
    parser.add_argument('--epochs', type=int, default=250, help='Num of epochs')
    parser.add_argument('--warmup', type=int, default=0, help='Num of warmup epochs') # ? delete
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--arch_learning_rate', type=float, default=0.0003, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=0.001, help='weight decay for arch encoding')
    parser.add_argument('--tau_min', type=float, default=10.0, help='The minimum tau for Gumbel')
    parser.add_argument('--tau_max', type=float, default=0.1, help='The maximum tau for Gumbel')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')
    parser.add_argument('--save_dir', type=str, help='Folder to save checkpoints and log.')
    parser.add_argument('--print_freq', type=int, default=200, help='print frequency (default: 200)')
    parser.add_argument('--rand_seed', type=int, default=1, help='manual seed')
    parser.add_argument('--fix_reduction', default=False, action='store_true',
                        help='Find or fix reduction cell')
    parser.add_argument('--mixed_prec', default=False, action='store_true',
                        help='Find or fix reduction cell')
    parser.add_argument('--deconv', default=False, action='store_true',
                        help='use deconvolution layer instead of standard convolution')
    parser.add_argument('--paper_arch', default=False, action='store_true',
                        help='use architecture in paper not in official implementation')
    parser.add_argument('--no_gumbel', default=False, action='store_true',
                        help='use gumbel softmax')

    xargs = parser.parse_args()
    xargs.num_classes = 10
    xargs.xshape = (1, 3, 32, 32)

    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.set_num_threads(xargs.workers)
    np.random.seed(xargs.rand_seed)
    torch.manual_seed(xargs.rand_seed)
    torch.cuda.manual_seed(xargs.rand_seed)
    torch.cuda.manual_seed_all(xargs.rand_seed)

    logger = prepare_logger(xargs)
    writer = SummaryWriter(xargs.save_dir + '/logs')

    train_data, valid_data, xshape, class_num = get_datasets(xargs.data_path, -1)
    search_loader = get_nas_search_loaders(train_data, './cifar-split.txt', xargs.batch_size, xargs.workers)

    search_space = ['none', 'skip_connect', 'dua_sepc_3x3', 'dua_sepc_5x5',
                    'dil_sepc_3x3', 'dil_sepc_5x5', 'avg_pool_3x3', 'max_pool_3x3']
    xargs.space = search_space

    search_model = get_cell_based_tiny_net(xargs)

    criterion = torch.nn.CrossEntropyLoss()
    w_optimizer = torch.optim.SGD(search_model.get_weights(), xargs.LR, momentum=xargs.momentum,
                                  weight_decay=xargs.decay, nesterov=xargs.nesterov)
    w_scheduler = CosineAnnealingLR(w_optimizer, T_max=len(search_loader) * xargs.epochs, eta_min=xargs.eta_min)
    a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999),
                                   weight_decay=xargs.arch_weight_decay)
    network, criterion = search_model.cuda(), criterion.cuda()

    model_base_path, model_best_path = logger.path('model'), logger.path('best')

    valid_accuracies = {'best': -1}
    genotypes = {-1: search_model.genotype()}
    if xargs.mixed_prec:
        network, [w_optimizer, a_optimizer] = amp.initialize(network, [w_optimizer, a_optimizer],
                                                             opt_level='O1')

    # start training
    total_epoch = xargs.epochs
    for epoch in tqdm(range(total_epoch)):
        search_model.set_tau(xargs.tau_max - (xargs.tau_max - xargs.tau_min) * epoch / (total_epoch - 1))
        valid_a_top1 = search_func(search_loader, network, criterion, w_scheduler, w_optimizer,
                                   a_optimizer, writer, epoch, mixed_prec=xargs.mixed_prec)

        # check the best accuracy
        valid_accuracies[epoch] = valid_a_top1
        genotypes[epoch] = search_model.genotype()

        # save checkpoint
        save_path = save_checkpoint({'epoch': epoch + 1,
                                     'args': deepcopy(xargs),
                                     'search_model': search_model.state_dict(),
                                     'w_optimizer': w_optimizer.state_dict(),
                                     'a_optimizer': a_optimizer.state_dict(),
                                     'w_scheduler': w_scheduler.state_dict(),
                                     'genotypes': genotypes,
                                     'valid_accuracies': valid_accuracies},
                                      model_base_path, logger)

        if valid_a_top1 > valid_accuracies['best']:
            valid_accuracies['best'] = valid_a_top1
            genotypes['best'] = search_model.genotype()
            copy_checkpoint(model_base_path, model_best_path, logger)

    with open(xargs.save_dir + '/args.json', 'w') as f:
        json.dump(xargs.__dict__, f, indent=2)

    logger.log('\n' + '-' * 100)
    logger.close()
