from .uhdm import uhdm_data_loader
from .fhdmi import fhdmi_data_loader
from .aim import aim_data_loader
from .tip import tip_data_loader
from .demo import demo_data_loader
from .utils import collate_to_device

import os
import torch.utils.data as data


def create_dataset(args, data_path, mode='train', device=None):
    if mode == 'demo':
        def _list_image_files_recursively(data_dir):
            file_list = []
            for home, dirs, files in os.walk(data_dir):
                for filename in files:
                    ext = filename.split(".")[-1]
                    if ext.lower() in ["jpg", "jpeg", "png", "gif", "webp"]:
                        file_list.append(os.path.join(home, filename))
            file_list.sort()
            return file_list
        data_files = _list_image_files_recursively(data_dir=data_path)
        dataset = demo_data_loader(data_files)
    elif args.DATA_TYPE == 'UHDM':
        def _list_image_files_recursively(data_dir):
            file_list = []
            for home, dirs, files in os.walk(data_dir):
                for filename in files:
                    if filename.endswith('gt.jpg'):
                        file_list.append(os.path.join(home, filename))
            file_list.sort()
            return file_list
        uhdm_files = _list_image_files_recursively(data_dir=data_path)
        dataset = uhdm_data_loader(args, uhdm_files, mode=mode)
    elif args.DATA_TYPE == 'FHDMi':
        fhdmi_files = sorted([file for file in os.listdir(os.path.join(data_path, 'target')) if file.endswith('.png')])
        dataset = fhdmi_data_loader(args, fhdmi_files, mode=mode)
    elif args.DATA_TYPE == 'TIP':
        tip_files = sorted([file for file in os.listdir(os.path.join(data_path, 'source')) if file.endswith('.png')])
        dataset = tip_data_loader(args, tip_files, mode=mode)
    elif args.DATA_TYPE == 'AIM':
        if mode == 'train':
            aim_files = sorted([file for file in os.listdir(os.path.join(data_path, 'moire')) if file.endswith('.jpg')])
        else:
            aim_files = sorted([file for file in os.listdir(os.path.join(data_path, 'moire')) if file.endswith('.png')])
        dataset = aim_data_loader(args, aim_files, mode=mode)
    else:
        raise NotImplementedError(f'Unrecognized DATA_TYPE: {args.DATA_TYPE}')

    use_ddp = 'RANK' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1
    if use_ddp:
        sampler = data.distributed.DistributedSampler(
            dataset,
            num_replicas=int(os.environ['WORLD_SIZE']),
            rank=int(os.environ['RANK']),
            shuffle=True,
            drop_last=True
        )
    else:
        sampler = None

    data_loader = data.DataLoader(
        dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=(sampler is None),
        num_workers=args.WORKER,
        drop_last=True,
        sampler=sampler,
        pin_memory=True,
        collate_fn=None
    )
    return data_loader
