import argparse, os
from PIL import ImageFile
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import json
from torch.utils.tensorboard import SummaryWriter

import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

ImageFile.LOAD_TRUNCATED_IMAGES = True  # ?

from procedures import prepare_logger, save_checkpoint, copy_checkpoint
from get_dataset_with_transform import get_datasets
from models import get_cifar_models
from train_proc import train_one_epoch
from train_proc import validate


def main(args):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.set_num_threads(args.workers)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed)

    logger = prepare_logger(args)

    train_data, valid_data, xshape, class_num = get_datasets(args.data_path, args.cutout_length)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)

    # base_model = obtain_model(model_config, args.extra_model_path)
    base_model = get_cifar_models(args)  ###########??????????????????????????????????????
    # optimizer, scheduler, criterion = get_optim_scheduler(base_model.parameters(), optim_config)
    optimizer = torch.optim.SGD(base_model.get_weights(), args.LR, momentum=args.momentum,
                                weight_decay=args.decay, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs,
                                                           eta_min=args.eta_min)
    criterion = torch.nn.CrossEntropyLoss()

    # last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
    model_base_path = args.model_path + f'/checkpoint/seed-{args.rand_seed}-basic.pth'
    model_best_path = args.model_path + f'/checkpoint/seed-{args.rand_seed}-best.pth'
    network, criterion = torch.nn.DataParallel(base_model).cuda(), criterion.cuda()

    # if last_info.exists():  # automatically resume from previous checkpoint
    #     logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    #     last_infox = torch.load(last_info)
    #     start_epoch = last_infox['epoch'] + 1
    #     last_checkpoint_path = last_infox['last_checkpoint']
    #     if not last_checkpoint_path.exists():
    #         logger.log('Does not find {:}, try another path'.format(last_checkpoint_path))
    #         last_checkpoint_path = last_info.parent / last_checkpoint_path.parent.name / last_checkpoint_path.name
    #     checkpoint = torch.load(last_checkpoint_path)
    #     base_model.load_state_dict(checkpoint['base-model'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     valid_accuracies = checkpoint['valid_accuracies']
    #     max_bytes = checkpoint['max_bytes']
    #     logger.log(
    #         "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
    # elif args.resume is not None:
    #     assert Path(args.resume).exists(), 'Can not find the resume file : {:}'.format(args.resume)
    #     checkpoint = torch.load(args.resume)
    #     start_epoch = checkpoint['epoch'] + 1
    #     base_model.load_state_dict(checkpoint['base-model'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     valid_accuracies = checkpoint['valid_accuracies']
    #     max_bytes = checkpoint['max_bytes']
    #     logger.log("=> loading checkpoint from '{:}' start with {:}-th epoch.".format(args.resume, start_epoch))
    # elif args.init_model is not None:
    #     assert Path(args.init_model).exists(), 'Can not find the initialization file : {:}'.format(args.init_model)
    #     checkpoint = torch.load(args.init_model)
    #     base_model.load_state_dict(checkpoint['base-model'])
    #     start_epoch, valid_accuracies, max_bytes = 0, {'best': -1}, {}
    #     logger.log('=> initialize the model from {:}'.format(args.init_model))
    # else:
    # logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch, valid_accuracies = 0, {'best': -1}

    # train_func, valid_func = get_procedures(args.procedure) ##########################

    total_epoch = args.epochs + args.warmup
    # Main Training and Evaluation Loop
    writer = SummaryWriter(args.save_dir + '/logs')

    for epoch in tqdm(range(start_epoch, total_epoch)):
        find_best = False
        # set-up drop-out ratio
        # if hasattr(base_model, 'update_drop_path'): #### ?????
        #     base_model.update_drop_path(
        #       args.drop_path_prob * epoch / total_epoch)

        # train for one epoch
        epoch_loss = train_one_epoch(network, train_loader, scheduler, optimizer, criterion, args, writer)
        writer.add_scalars('Loss', {'Train_loss': epoch_loss}, epoch)
        # evaluate the performance
        if (epoch % args.eval_frequency == 0) or (epoch + 1 == total_epoch):
            valid_acc1, valid_loss = validate(network, valid_loader, criterion, args, writer)
            writer.add_scalars('Loss', {'Val_loss': valid_loss}, epoch)
            writer.add_scalars('Accuracy', {'Accuracy': valid_acc1}, epoch)

            valid_accuracies[epoch] = valid_acc1
            if valid_acc1 > valid_accuracies['best']:
                valid_accuracies['best'] = valid_acc1
                torch.save({
                    'epoch': epoch,
                    'base-model': base_model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, model_best_path)

        if epoch % 10 == 0:
            torch.cuda.empty_cache()

        # save checkpoint
        # save_path = save_checkpoint({
        #     'epoch': epoch,
        #     'valid_accuracies': deepcopy(valid_accuracies),
        #     'base-model': base_model.state_dict(),
        #     'scheduler': scheduler.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }, model_base_path, logger)
        torch.save({
            'epoch': epoch,
            'base-model': base_model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_base_path)

        #     copy_checkpoint(model_base_path, model_best_path, logger)
        # last_info = save_checkpoint({
        #     'epoch': epoch,
        #     'args': deepcopy(args),
        #     'last_checkpoint': save_path,
        # }, logger.path('info'), logger)

    logger.log('-' * 200 + '\n')
    logger.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train a classification model on typical image classification datasets.')
    # parser.add_argument('--resume', type=str, help='Resume path.')
    # parser.add_argument('--init_model', type=str, help='The initialization model path.')
    parser.add_argument('--ichannel', type=int, default=33)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--stem_multi', type=int, default=3)
    parser.add_argument('--drop_path_prob', type=float, default=0.2)
    # "super_type": ["str", "infer-nasnet.cifar"],
    # "genotype": ["none", "none"],
    parser.add_argument('--scheduler', type=str, default='cos')
    parser.add_argument('--eta_min', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=295)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--LR', type=float, default=0.025)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--criterion', type=str, default='Softmax')
    parser.add_argument('--auxiliary', type=float, default=0.4)

    # parser.add_argument('--model_config', type=str, help='The path to the model configuration')
    # parser.add_argument('--optim_config', type=str, help='The path to the optimizer configuration')
    parser.add_argument('--procedure', type=str, default='basic', help='The procedure basic prefix.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to search algorithm generated part')
    # Data Generation
    # parser.add_argument('--dataset', type=str, help='The dataset name.')
    parser.add_argument('--data_path', type=str, required=True, help='The dataset name.')
    parser.add_argument('--cutout_length', type=int, default=16, help='The cutout length, negative means not use.')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency (default: 200)')
    parser.add_argument('--print_freq_eval', type=int, default=100, help='print frequency (default: 200)')
    parser.add_argument('--eval_frequency', type=int, default=1, help='evaluation frequency (default: 200)')
    parser.add_argument('--save_dir', type=str, required=True, help='Folder to save checkpoints and log.')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers (default: 8)')
    # Random Seed
    parser.add_argument('--rand_seed', type=int, default=1, help='manual seed')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    args = parser.parse_args()

    if not os.isfile(args.model_path):
        raise ValueError('invalid model_path : {:}'.format(args.model_path))

    main(args)
    with open(args.save_dir + '/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
