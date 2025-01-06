import argparse
import datetime
import json
import math
import os
import sys
import time
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.lr_sched as lr_sched
import util.misc as misc
from data.dataset import IMUSyncDataset
from models.units import UniTS, UniTSArgs
from util.misc import NativeScalerWithGradNormCount as NativeScaler

def get_args_parser():
    parser = argparse.ArgumentParser('UniTS awaretraining', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing effective batch size)')

    # Model parameters
    parser.add_argument('--d_model', default=128, type=int)
    parser.add_argument('--n_heads', default=8, type=int)
    parser.add_argument('--e_layers', default=3, type=int)
    parser.add_argument('--patch_len', default=8, type=int)
    parser.add_argument('--stride', default=8, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--prompt_num', default=10, type=int)
    
    # Pretrain parameters
    parser.add_argument('--min_mask_ratio', default=0.7, type=float)
    parser.add_argument('--max_mask_ratio', default=0.8, type=float)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N')

    # Dataset parameters
    parser.add_argument('--data_config', default=None, help='dataset config path')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Training parameters
    parser.add_argument('--output_dir', default='./output_pretrain')
    parser.add_argument('--log_dir', default='./output_pretrain')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # train setting
    parser.add_argument('--setting_id', default=0, type=int, help='training setting')
    parser.add_argument('--phase', default='all', type=str, help='all, cls')
    # deprecated, always enable aware layer
    parser.add_argument('--enable_aware', action='store_true', help='enable aware layer')
    parser.set_defaults(enable_aware=False)
    return parser

# def calculate_contrastive_loss(x, y, temperature=0.1):
#     """
#     x: Final feature tensor for first augmented view [batch_size, d_model]
#     y: Final feature tensor for second augmented view [batch_size, d_model]
#     temperature: Temperature parameter for scaling logits
#     """
#     # Normalize the features
#     x = F.normalize(x, dim=1)
#     y = F.normalize(y, dim=1)
    
#     # Compute similarity matrix between positive pairs
#     batch_size = x.size(0)
#     labels = torch.arange(batch_size, device=x.device)
    
#     # Compute logits
#     logits_xy = torch.matmul(x, y.t()) / temperature
#     logits_yx = torch.matmul(y, x.t()) / temperature
    
#     # Compute loss for both directions
#     loss_x = F.cross_entropy(logits_xy, labels)
#     loss_y = F.cross_entropy(logits_yx, labels)
    
#     return (loss_x + loss_y) / 2

def calculate_clip_loss(x, y, temperature=0.07):
    """
    Args:
        x (torch.Tensor): Feature tensor for first modality, cls token, shape [batch_size, d_model]
        y (torch.Tensor): Feature tensor for second modality, cls token, shape [batch_size, d_model]
        temperature (float): Temperature parameter for scaling logits

    Returns:
        torch.Tensor: Scalar CLIP-like contrastive loss
    """
    
    # Normalize features to unit vectors
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    
    # Compute pairwise similarity: [batch_size, batch_size]
    logits_xy = torch.matmul(x, y.t()) / temperature
    
    # By symmetry, we can reuse the same matrix transposed for the other direction
    # or simply compute y -> x as well (equivalent to logits_xy.t() if shapes match).
    logits_yx = logits_xy.t()
    
    # Labels: each sample in the batch is the "positive" for itself
    batch_size = x.size(0)
    labels = torch.arange(batch_size, device=x.device)
    
    # Cross-entropy for x->y
    loss_xy = F.cross_entropy(logits_xy, labels)
    # Cross-entropy for y->x
    loss_yx = F.cross_entropy(logits_yx, labels)
    
    # Final CLIP-like loss is the average of the two directions
    return (loss_xy + loss_yx) / 2

def train_one_epoch(model: nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, 
                    loss_scaler,
                    log_writer=None,
                    args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, (_, imu_input, location_emb, _, sync_input, sync_location_emb) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        imu_input = imu_input.to(device, non_blocking=True)
        sync_input = sync_input.to(device, non_blocking=True)
        location_emb = location_emb.to(device, non_blocking=True)
        sync_location_emb = sync_location_emb.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            x, y = model(imu_input, prior_emb=location_emb, y=sync_input, prior_y=sync_location_emb)
            loss = calculate_clip_loss(x, y)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        #     epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        #     log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
        #     log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(model: nn.Module,
                   data_loader: Iterable, 
                   device: torch.device, 
                   epoch: int, 
                   log_writer=None,
                   args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}] Vali:'
    print_freq = 10

    with torch.no_grad():
        for _, imu_input, location_emb, _, sync_input, sync_location_emb in metric_logger.log_every(data_loader, print_freq, header):
            imu_input = imu_input.to(device, non_blocking=True)
            sync_input = sync_input.to(device, non_blocking=True)
            location_emb = location_emb.to(device, non_blocking=True)
            sync_location_emb = sync_location_emb.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                mask_out, mask_seq = model(imu_input, prior_emb=location_emb, prior_y=sync_location_emb)
                loss = calculate_reconstruction_loss(sync_input, mask_out, mask_seq)

            loss_value = loss.item()
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                    print("Loss is {}, stopping evaluation".format(loss_value))
                    sys.exit(1)
        
            metric_logger.update(loss=loss_value)

    metric_logger.synchronize_between_processes()
    print("Averaged stats for Vali:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    misc.init_distributed_mode(args)   
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Save args
    model_args = UniTSArgs.from_args(args)
    log_args = {
        'time': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'train_args': vars(args),
        'model_args': vars(model_args),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), mode="w", encoding="utf-8") as f:
        f.write(json.dumps(log_args, indent=4) + "\n")

    # Create dataset
    dataset_train = IMUSyncDataset(args.data_config, is_train=True, is_rotated=args.setting_id == 1)
    print(f"train dataset size: {len(dataset_train)}")

    # Split into train and validation sets
    val_size = len(dataset_train) // 9  # 10% for validation
    train_size = len(dataset_train) - val_size
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset_train, [train_size, val_size], generator=generator
    )

    # Create samplers for distributed training
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )

    # Create data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Define the model
    model = UniTS(
        enc_in=6,  # 6 channels for IMU data (3 acc + 3 gyro)
        num_class=7,  # Number of activity classes
        args=args,
        task='aware'
    )
    model.to(device)

    # Set up distributed training
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # Print trainable parameters
    print("Trainable parameters:")
    print([(key, val.shape) for key, val in model.named_parameters() if val.requires_grad])

    # Calculate effective batch size
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    # Set learning rate
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # Set up optimizer
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # Set up tensorboard logging
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Training loop
    print(f"Start pretraining for {args.epochs} epochs")
    start_time = time.time()
    
    lowest_vali_epoch = 0
    lowest_vali_loss = float('inf')
    span = 20

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            data_loader_val.sampler.set_epoch(epoch)

        # Train for one epoch
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # Evaluate
        val_stats = evaluate(
            model, data_loader_val,
            device, epoch, log_writer=log_writer,
            args=args
        )

        # Save checkpoint
        if args.output_dir:
            is_save = False
            if lowest_vali_loss > val_stats['loss']:
                lowest_vali_loss = val_stats['loss']
                lowest_vali_epoch = epoch
                is_save = True
            if (epoch + 1) % span == 0 or epoch + 1 == args.epochs:
                is_save = True
            if is_save:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch
                )

        # Log stats
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'epoch': epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "best.json"), mode="w", encoding="utf-8") as f:
            f.write(json.dumps({
                "lowest_vali_epoch": lowest_vali_epoch, 
                "lowest_vali_loss": lowest_vali_loss,
            }, indent=4) + "\n")

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)