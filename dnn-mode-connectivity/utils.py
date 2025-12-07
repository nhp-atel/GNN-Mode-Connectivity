import numpy as np
import os
import torch
import torch.nn.functional as F

import curves


def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()

    # Detect device from model parameters
    device = next(model.parameters()).device

    for iter, batch in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)

        # Handle both graph batches (PyG) and image batches (standard)
        if hasattr(batch, 'x'):  # Graph batch
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y
            batch_size = batch.num_graphs
        else:  # Image batch (backward compatibility)
            input, target = batch
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            batch_size = input.size(0)

        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * batch_size
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()

    # Detect device from model parameters
    device = next(model.parameters()).device

    for batch in test_loader:
        # Handle both graph batches (PyG) and image batches (standard)
        if hasattr(batch, 'x'):  # Graph batch
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch, **kwargs)
            target = batch.y
            batch_size = batch.num_graphs
        else:  # Image batch (backward compatibility)
            input, target = batch
            input = input.to(device)
            target = target.to(device)
            output = model(input, **kwargs)
            batch_size = input.size(0)

        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * batch_size
        loss_sum += loss.item() * batch_size
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }



def predictions(test_loader, model, **kwargs):
    model.eval()

    # Detect device from model parameters
    device = next(model.parameters()).device

    preds = []
    targets = []
    for batch in test_loader:
        # Handle both graph batches (PyG) and image batches (standard)
        if hasattr(batch, 'x'):  # Graph batch
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch, **kwargs)
            target = batch.y
        else:  # Image batch (backward compatibility)
            input, target = batch
            input = input.to(device)
            output = model(input, **kwargs)

        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.cpu().numpy() if hasattr(batch, 'x') else target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()

    # Detect device from model parameters
    device = next(model.parameters()).device

    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for batch in loader:
        # Handle both graph batches (PyG) and image batches (standard)
        if hasattr(batch, 'x'):  # Graph batch
            batch = batch.to(device)
            batch_size = batch.num_graphs
            momentum = batch_size / (num_samples + batch_size)
            for module in momenta.keys():
                module.momentum = momentum
            model(batch.x, batch.edge_index, batch.batch, **kwargs)
        else:  # Image batch (backward compatibility)
            input, _ = batch
            input = input.to(device)
            batch_size = input.data.size(0)
            momentum = batch_size / (num_samples + batch_size)
            for module in momenta.keys():
                module.momentum = momentum
            model(input, **kwargs)

        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))
