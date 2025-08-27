# utils/runner.py
import os
import shutil
from typing import Dict, Tuple, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.common import mkdir


# -----------------------------
# DDP lifecycle
# -----------------------------
def init_ddp(args) -> Tuple[torch.device, int, int]:
    """Initialize DistributedDataParallel (NCCL)."""
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    dist.init_process_group("nccl")
    return device, rank, world_size


def finalize_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# -----------------------------
# Experiment directories
# -----------------------------
def setup_train_dirs(args, rank: int):
    """
    Prepare experiment directories for TRAIN:
        <SAVE_PREFIX>/<EXP_NAME>/{logs, net_checkpoints, train_visual}
    If START_FRESH=True and RESUME='none', remove previous experiment dir.
    """
    exp_root = os.path.join(args.SAVE_PREFIX, args.EXP_NAME)
    args.EXP_ROOT = exp_root
    args.LOGS_DIR = os.path.join(exp_root, "logs")
    args.NETS_DIR = os.path.join(exp_root, "net_checkpoints")
    args.VISUALS_DIR = os.path.join(exp_root, "train_visual")

    if rank == 0:
        if getattr(args, "START_FRESH", False) and getattr(args, "RESUME", "none") == "none":
            shutil.rmtree(exp_root, ignore_errors=True)
        mkdir(args.LOGS_DIR)
        mkdir(args.NETS_DIR)
        mkdir(args.VISUALS_DIR)


def setup_test_dirs(args, rank: int):
    """
    Prepare directories for TEST:
        <SAVE_PREFIX>/<EXP_NAME>/{test_result, net_checkpoints}
    """
    args.TEST_RESULT_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, "test_result")
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, "net_checkpoints")
    if rank == 0:
        mkdir(args.TEST_RESULT_DIR)


# -----------------------------
# Checkpoint helpers (train)
# -----------------------------
def ckpt_path(args, epoch: Optional[int] = None, latest: bool = False) -> str:
    if latest:
        return os.path.join(args.NETS_DIR, "checkpoint_latest.tar")
    assert epoch is not None, "epoch must be specified unless latest=True"
    return os.path.join(args.NETS_DIR, f"checkpoint_{epoch:06d}.tar")


def save_checkpoint(args, epoch, iters, model, optimizer, scheduler):
    """Save latest checkpoint and periodic checkpoints. Rank 0 only."""
    if dist.get_rank() != 0:
        return
    state = {
        "epoch": epoch,
        "iters": iters,
        "state_dict": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "learning_rate": optimizer.param_groups[0]["lr"],
    }
    # latest
    torch.save(state, ckpt_path(args, latest=True))
    # periodic
    if epoch % getattr(args, "SAVE_EVERY", 10) == 0:
        torch.save(state, ckpt_path(args, epoch=epoch))


def resume_from_checkpoint(args, model, optimizer, scheduler) -> Tuple[int, int]:
    """
    Load checkpoint according to RESUME policy:
      - RESUME in {'none','latest','epoch','path'}
      - (legacy) if LOAD_EPOCH given -> RESUME=epoch
    Returns: (start_epoch, iters)
    """
    # Legacy compatibility
    if getattr(args, "LOAD_EPOCH", None):
        args.RESUME = "epoch"
        args.RESUME_EPOCH = args.LOAD_EPOCH

    mode = getattr(args, "RESUME", "none")
    if mode == "none":
        return 0, 0

    if mode == "latest":
        path = ckpt_path(args, latest=True)
    elif mode == "epoch":
        ep = int(getattr(args, "RESUME_EPOCH", 0))
        path = ckpt_path(args, epoch=ep)
    elif mode == "path":
        path = getattr(args, "RESUME_PATH", None)
    else:
        raise ValueError(f"Invalid RESUME: {mode}")

    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu")

    # Load model
    sd = ckpt["state_dict"]
    if isinstance(model, DDP):
        model.module.load_state_dict(sd)
    else:
        model.load_state_dict(sd)

    # Load optimizer & scheduler
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch = int(ckpt.get("epoch", 0))
    iters = int(ckpt.get("iters", 0))

    if dist.get_rank() == 0:
        print(f"[Resume] Loaded checkpoint: {path} | epoch={start_epoch} | iters={iters}")
    return start_epoch, iters


# -----------------------------
# Checkpoint helpers (test)
# -----------------------------
def resolve_test_checkpoint(args, rank: int) -> Tuple[str, str]:
    """
    Decide checkpoint path and output directory for TEST.
      - If LOAD_PATH given: use it, save_dir = test_result
      - Else if TEST_EPOCH == 'auto': use latest,   save_dir = test_result/latest
      - Else: use specific epoch,                   save_dir = test_result/<ep:04d>
    Returns: (ckpt_path, save_dir)
    """
    if getattr(args, "LOAD_PATH", None):
        load_path = args.LOAD_PATH
        save_dir = args.TEST_RESULT_DIR
    else:
        ep = args.TEST_EPOCH
        if ep == "auto":
            load_path = os.path.join(args.NETS_DIR, "checkpoint_latest.tar")
            save_dir = os.path.join(args.TEST_RESULT_DIR, "latest")
        else:
            ep = int(ep)
            load_path = os.path.join(args.NETS_DIR, f"checkpoint_{ep:06d}.tar")
            save_dir = os.path.join(args.TEST_RESULT_DIR, f"{ep:04d}")

    if rank == 0:
        mkdir(save_dir)
    return load_path, save_dir


def load_model_weights(model, ckpt_path: str):
    """Load weights into (possibly DDP-wrapped) model."""
    if ckpt_path.endswith(".pth"):
        sd = torch.load(ckpt_path, map_location="cpu")
    else:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    if isinstance(model, DDP):
        model.module.load_state_dict(sd)
    else:
        model.load_state_dict(sd)


# -----------------------------
# Misc helpers
# -----------------------------
def to_device(batch: Dict, device: torch.device) -> Dict:
    """Move all tensors in a dict batch to device (non_blocking=True)."""
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
