# train.py
import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torchvision

from models import create_model
from datasets import create_dataset
from losses import create_loss
from utils.common import set_seed
from config.config import get_parser

# <-- 공통 러너 유틸 -->
from utils.runner import (
    init_ddp, finalize_ddp,
    setup_train_dirs,
    save_checkpoint, resume_from_checkpoint,
    to_device
)

# ----------------------------------
# Visualization helper
# ----------------------------------
def save_visuals(args, iters, in_img, out_img, gt_img):
    if dist.get_rank() != 0:
        return
    if iters % args.SAVE_ITER != (args.SAVE_ITER - 1):
        return
    grid = torch.cat((in_img.detach().cpu(),
                      out_img.detach().cpu(),
                      gt_img.detach().cpu()), dim=3)
    save_num = (iters + 1) // args.SAVE_ITER
    path = os.path.join(args.VISUALS_DIR, f"visual_x{args.SAVE_ITER:04d}_{save_num:05d}.jpg")
    torchvision.utils.save_image(grid, path)

# ----------------------------------
# Train primitives
# ----------------------------------
def train_step(args, batch, model, loss_fn, optimizer, iters):
    model.train()
    outputs = model(batch)                 # [full, half, quarter]
    loss = loss_fn(outputs, batch['label'])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # save only the first scale visualization
    save_visuals(args, iters, batch['in_img'], outputs[0], batch['label'])
    return loss.item()

def train_epoch(args, loader, model, loss_fn, optimizer, scheduler, device, epoch, iters):
    model.train()
    bar = tqdm(loader, disable=(dist.get_rank() != 0))
    total = 0.0
    lr = optimizer.param_groups[0]['lr']

    for idx, batch in enumerate(bar):
        batch = to_device(batch, device)
        loss_val = train_step(args, batch, model, loss_fn, optimizer, iters)
        iters += 1
        total += loss_val
        avg = total / (idx + 1)
        if dist.get_rank() == 0:
            bar.set_description(f"Epoch {epoch} | lr {lr:.7f} | loss {avg:.5f}")

    scheduler.step()
    return lr, avg, iters

# ----------------------------------
# Main
# ----------------------------------
def main():
    args = get_parser()

    # DDP init
    device, rank, world_size = init_ddp(args)

    # exp dirs (TensorBoard/logs, ckpts, visuals)
    setup_train_dirs(args, rank)

    # (optional legacy) LOAD_EPOCH -> RESUME=epoch 로 해석은 runner.resume_from_checkpoint 내부에서 처리

    set_seed(args.SEED)

    # Data
    TrainImgLoader = create_dataset(args, data_path=args.TRAIN_DATASET, mode='train', device=device)

    # Model / Optim / Sched / Loss
    model = create_model(args).to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    optimizer = optim.Adam(
        [{'params': model.module.parameters(), 'initial_lr': args.BASE_LR}],
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, T_mult=args.T_MULT, eta_min=args.ETA_MIN
    )
    loss_fn = create_loss(args).to(device)

    # Resume (latest/epoch/path/none)
    start_epoch, iters = resume_from_checkpoint(args, model, optimizer, scheduler)

    # TensorBoard only (no .log file)
    writer = SummaryWriter(args.LOGS_DIR) if rank == 0 else None


    # Train loop
    for epoch in range(start_epoch + 1, args.EPOCHS + 1):
        if hasattr(TrainImgLoader, 'sampler') and TrainImgLoader.sampler is not None:
            TrainImgLoader.sampler.set_epoch(epoch)

        lr, avg_loss, iters = train_epoch(
            args, TrainImgLoader, model, loss_fn, optimizer, scheduler, device, epoch, iters
        )

        if rank == 0 and writer:
            writer.add_scalar('Train/avg_loss', avg_loss, epoch)
            writer.add_scalar('Train/learning_rate', lr, epoch)

        # Save ckpt: latest + periodic(SAVE_EVERY)
        save_checkpoint(args, epoch, iters, model, optimizer, scheduler)

    if writer:
        writer.close()
    finalize_ddp()

if __name__ == '__main__':
    main()
