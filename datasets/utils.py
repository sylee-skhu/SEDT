# dataset/dataset.py

import os
import torch
import torch.utils.data as data


def collate_to_device(batch, device):
    """
    배치가 [{'in_img':..., 'label':...}, ...] 꼴로 오면,
    각 항목을 모아서 device에 올려줌
    """
    batch_dict = {}
    for key in batch[0]:
        vals = [d[key] for d in batch]
        if torch.is_tensor(vals[0]):
            batch_dict[key] = torch.stack(vals).to(device, non_blocking=True)
        else:
            batch_dict[key] = vals
    return batch_dict


def default_loader(path_set=[]):
    from PIL import Image
    from torchvision import transforms
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        img = transforms.ToTensor()(img)
        imgs.append(img)
    return imgs


def crop_loader(crop_size, x, y, path_set=[]):
    from PIL import Image
    from torchvision import transforms
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        img = img.crop((x, y, x + crop_size, y + crop_size))
        img = transforms.ToTensor()(img)
        imgs.append(img)
    return imgs


def resize_loader(resize_size, path_set=[]):
    from PIL import Image
    from torchvision import transforms
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        img = img.resize((resize_size, resize_size), Image.BICUBIC)
        img = transforms.ToTensor()(img)
        imgs.append(img)
    return imgs


def default_toTensor(img):
    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)
    return composed_transform(img)
