import argparse
import json
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from get_dataset_with_transform import get_datasets
from models import NASNetonCIFAR
from train_proc import train_one_epoch, validate

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def main(args):
    # get datasets and dataloaders
    train_data, valid_data, xshape, class_num = get_datasets(args.data_path, args.cutout_length)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)
    # load searched model genotype, infer the network
    xdata = torch.load(args.model_path)
    genotype = xdata['genotypes'][xdata['epoch'] - 1]
    with open('/'.join(args.model_path.split('/')[:-2]) + '/args.json', 'r') as file:
        search_args = json.load(file)
    base_model = NASNetonCIFAR(args.ichannel, args.layers, args.stem_multi, args.class_num, genotype, args.auxiliary,
                               paper_arch=search_args['paper_arch'], fix_reduction=search_args['fix_reduction'])
    # scheduler, optimizer and loss
    optimizer = torch.optim.SGD(base_model.parameters(), args.LR, momentum=args.momentum,
                                weight_decay=args.decay, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    criterion = torch.nn.CrossEntropyLoss()
    # save paths
    model_base_path = args.save_dir + f'/checkpoint/seed-{args.rand_seed}-basic.pth'
    model_best_path = args.save_dir + f'/checkpoint/seed-{args.rand_seed}-best.pth'

    network, criterion = base_model.cuda(), criterion.cuda()

    best_val_acc = -1
    writer = SummaryWriter(args.save_dir + '/logs')
    # training part

    for epoch in tqdm(range(args.epochs)):
        # set-up drop-out ratio
        # if hasattr(base_model, 'update_drop_path'): #### ?????
        #     base_model.update_drop_path(
        #       args.drop_path_prob * epoch / total_epoch)

        # train for one epoch
        train_one_epoch(network, train_loader, optimizer, criterion, args, writer, epoch)
        # evaluate the performance
        valid_acc1, valid_loss = validate(network, valid_loader, criterion)
        # write into tensorboard
        writer.add_scalars('Loss', {'Val_loss': valid_loss}, epoch)
        writer.add_scalars('Accuracy', {'Accuracy': valid_acc1}, epoch)
        writer.add_scalars('Learning_rate', {'lr': optimizer.state_dict()['param_groups'][0]['lr']}, epoch)
        scheduler.step()
        # save current model
        torch.save({
            'epoch': epoch,
            'base-model': base_model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_base_path)
        # if found better model, save as best
        if valid_acc1 > best_val_acc:
            best_val_acc = valid_acc1
            torch.save({
                'epoch': epoch,
                'base-model': base_model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_best_path)
    if epoch % 10 == 0:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a classification model on typical image classification datasets.')
    parser.add_argument('--ichannel', type=int, default=36)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--stem_multi', type=int, default=3)
    # parser.add_argument('--drop_path_prob', type=float, default=0.2)
    parser.add_argument('--eta_min', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--LR', type=float, default=0.025)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--criterion', type=str, default='Softmax')
    parser.add_argument('--auxiliary', type=float, default=0.4)
    parser.add_argument('--model_path', type=str, required=True, help='Path to search algorithm generated file')
    parser.add_argument('--data_path', type=str, required=True, help='The dataset path.')
    parser.add_argument('--cutout_length', type=int, default=16, help='The cutout length, negative means not use.')
    parser.add_argument('--save_dir', type=str, required=True, help='Folder to save checkpoints and log.')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--rand_seed', type=int, default=1, help='manual seed')
    parser.add_argument('--batch_size', type=int, default=96, help='Batch size for training.')
    parser.add_argument('--paper_arch', default=False, action='store_true',
                        help='Use architecture from paper or from official implementation')
    args = parser.parse_args()
    args.class_num = 10
    if not os.path.isfile(args.model_path):
        raise ValueError('invalid model_path : {:}'.format(args.model_path))

    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.set_num_threads(args.workers)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed)

    main(args)
    with open(args.save_dir + '/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
