import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import torchvision
from torchvision.utils import save_image, make_grid
from models import create_model
from datasets import create_dataset
from utils.common import set_seed, pad_and_replace, mkdir
from utils.metric import create_metrics
from config.config import get_parser

# runner 모듈화 유틸
from utils.runner import (
    init_ddp,
    finalize_ddp,
    setup_test_dirs,
    resolve_test_checkpoint,
    load_model_weights,
    to_device,
)


# -----------------------------
# Test primitives
# -----------------------------
@torch.no_grad()
def test_step(args, batch, model, device, save_dir, compute_metrics):
    """
    - 추론 시간 측정
    - padding 제거
    - metric 계산(옵션)
    - 이미지 저장 (rank0 & args.SAVE_IMG)
    """
    number = batch["number"]
    label = batch["label"]
    in_img = batch["in_img"]

    fname = number[0] if isinstance(number, (list, tuple)) else number

    # pad/replace
    data_mod, h_pad, h_odd_pad, w_pad, w_odd_pad = pad_and_replace(batch)

    # 추론 시간 측정
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = model(data_mod)[0]
    end.record()
    torch.cuda.synchronize()
    cur_time = start.elapsed_time(end) / 1000.0  # sec

    # depad
    if h_pad != 0:
        out = out[:, :, h_pad:-h_odd_pad, :]
        label = label[:, :, h_pad:-h_odd_pad, :]
        in_img = in_img[:, :, h_pad:-h_odd_pad, :]
    if w_pad != 0:
        out = out[:, :, :, w_pad:-w_odd_pad]
        label = label[:, :, :, w_pad:-w_odd_pad]
        in_img = in_img[:, :, :, w_pad:-w_odd_pad]

    # metrics
    lpips_v, psnr_v, ssim_v = compute_metrics.compute(out, label)

    # --- 저장 (rank0 & args.SAVE_IMG) : in/gt/out 저장
    if dist.get_rank() == 0 and args.SAVE_IMG:
        ext = args.SAVE_IMG if isinstance(args.SAVE_IMG, str) else "png"
        img_dir = os.path.join(save_dir, "images")
        mkdir(img_dir)

        in_cpu  = in_img.detach().cpu()
        gt_cpu  = label.detach().cpu()
        out_cpu = out.detach().cpu()

        save_image(in_cpu,  os.path.join(img_dir, f"{fname}_in.{ext}"))
        save_image(gt_cpu,  os.path.join(img_dir, f"{fname}_gt.{ext}"))
        save_image(out_cpu, os.path.join(img_dir, f"{fname}_out.{ext}"))

    return psnr_v, ssim_v, lpips_v, cur_time


def run_test(args, loader, model, device, save_dir, compute_metrics):
    bar = tqdm(loader, disable=(dist.get_rank() != 0))
    tot_psnr = tot_ssim = tot_lpips = tot_time = 0.0

    for idx, batch in enumerate(bar):
        batch = to_device(batch, device)
        model.eval()

        psnr_v, ssim_v, lpips_v, tsec = test_step(args, batch, model, device, save_dir, compute_metrics)

        tot_psnr += psnr_v
        tot_ssim += ssim_v
        tot_lpips += lpips_v

        if idx > 5:  # warm-up 제외
            tot_time += tsec
            avg_time = tot_time / (idx - 5)
        else:
            avg_time = 0.0

        avg_psnr = tot_psnr / (idx + 1)
        avg_ssim = tot_ssim / (idx + 1)
        avg_lpips = tot_lpips / (idx + 1)

        if dist.get_rank() == 0:
            bar.set_description(
                f"LPIPS={avg_lpips:.4f} | PSNR={avg_psnr:.4f} | SSIM={avg_ssim:.4f} | Avg.TIME={avg_time:.4f}s"
            )

    if dist.get_rank() == 0:
        print(f"[Test] Avg LPIPS={avg_lpips:.4f} | PSNR={avg_psnr:.4f} | SSIM={avg_ssim:.4f}")
        print(f"[Test] Avg TIME={avg_time:.4f}s")


# -----------------------------
# Main
# -----------------------------
def main():
    args = get_parser()

    # DDP init (runner)
    device, rank, world_size = init_ddp(args)

    # 디렉토리 준비 (runner)
    setup_test_dirs(args, rank)

    set_seed(args.SEED)

    # Data
    args.BATCH_SIZE = 1  # 고정
    test_loader = create_dataset(args, data_path=args.TEST_DATASET, mode="test", device=device)

    # Model
    model = create_model(args).to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Checkpoint 결정/로드 (runner)
    ckpt_path, save_dir = resolve_test_checkpoint(args, rank)
    load_model_weights(model, ckpt_path)
    if rank == 0:
        print(f"[Test] Load weights: {ckpt_path}")
        print(f"[Test] Save outputs to: {save_dir}")

    compute_metrics = create_metrics(args, device=device)

    # Run
    run_test(args, test_loader, model, device, save_dir, compute_metrics)

    # DDP finalize (runner)
    finalize_ddp()


if __name__ == "__main__":
    main()