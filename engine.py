

import torch
import torch.nn as nn
from collections.abc import Iterable
import math
import sys

import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(model: nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    # model.module.set_default_trainability()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    correct = 0
    total = 0

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (label, imu_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        criterion = torch.nn.CrossEntropyLoss()
        imu_input = imu_input.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            output = model(imu_input)
            c_loss = criterion(output, label.long().squeeze(-1))

        # Calculate accuracy
        _, preds = torch.max(output, 1) # output: [B, 8]
        label_indices = label.view(-1) # label: [B, 1]
        correct += (preds == label_indices).sum().item()
        total += label.size(0)
        

        # Loss
        loss = c_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        m_loss_value = c_loss

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(mloss=m_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        m_loss_value_reduce = misc.all_reduce_mean(m_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('m_train_loss', m_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # Compute accuracy
    accuracy = 100. * correct / total
    metric_logger.meters['acc'].update(accuracy)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats for Train:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model, data_loader, device, epoch, args=None, is_test=False):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}] {"Test" if is_test else "Validation"}:'
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 10, header):
            targets, images = batch
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Compute output
            outputs = model(images)
            loss = criterion(outputs, targets.long().squeeze(-1))

            # Measure accuracy
            _, preds = torch.max(outputs, 1) # output: [B, 8]
            label_indices = targets.view(-1) # label: [B, 1]
            correct += (preds == label_indices).sum().item()
            total += targets.size(0)

            # Update metrics
            metric_logger.update(loss=loss.item())

    # Compute accuracy
    accuracy = 100. * correct / total
    metric_logger.meters['acc'].update(accuracy)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats for {'Test' if is_test else 'Validation'}:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
