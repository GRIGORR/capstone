import torch
import numpy as np


def train_one_epoch(network, data_loader, optimizer, criterion, args, writer, epoch):
    network.train()
    for i, (inputs, targets) in enumerate(data_loader):
        targets = targets.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)
        optimizer.zero_grad()
        features, logits = network(inputs)
        if isinstance(logits, list):
            assert len(logits) == 2, 'logits must has {:} items instead of {:}'.format(2, len(logits))
            logits, logits_aux = logits
        else:
            logits, logits_aux = logits, None
        loss = criterion(logits, targets)
        if args.auxiliary > 0:
            loss_aux = criterion(logits_aux, targets)
            loss += args.auxiliary * loss_aux
        writer.add_scalars('Loss', {'Train_loss': loss}, epoch*len(data_loader) + i)
        loss.backward()
        optimizer.step()


def validate(network, data_loader, criterion):
    network.eval()
    accuracy = []
    val_loss = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            targets = targets.cuda(non_blocking=True)
            inputs = inputs.cuda(non_blocking=True)

            features, logits = network(inputs)
            loss = criterion(logits, targets)
            val_loss.append(loss.item())
            _, preds = torch.max(logits, 1)
            accuracy.append((torch.sum(preds == targets).item() / targets.shape[0]) * 100)

    return np.mean(accuracy), np.mean(val_loss)


def _parse(weights, steps, space, edge_ids=[0, 1]):
    gene = []
    edge2indx = {'0<-0': 0, '0<-1': 1, '1<-0': 2, '1<-1': 3, '1<-2': 4, '2<-0': 5, '2<-1': 6, '2<-2': 7,
                 '2<-3': 8, '3<-0': 9, '3<-1': 10, '3<-2': 11, '3<-3': 12, '3<-4': 13}
    for i in range(steps):
        edges = []
        for j in range(2 + i):
            node_str = '{:}<-{:}'.format(i, j)
            ws = weights[edge2indx[node_str]]
            for k, op_name in enumerate(space):
                if op_name == 'none':
                    continue
                edges.append((op_name, j, ws[k]))
        edges = sorted(edges, key=lambda x: -x[-1])
        selected_edges = edges[edge_ids[0]], edges[edge_ids[1]]
        gene.append(tuple(selected_edges))
    return gene