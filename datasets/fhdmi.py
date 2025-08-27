import os
import random
import torch.utils.data as data
from PIL import ImageFile
from .utils import crop_loader, resize_loader, default_loader


class fhdmi_data_loader(data.Dataset):
    def __init__(self, args, image_list, mode='train'):
        self.image_list = image_list
        self.args = args
        self.mode = mode
        if mode == 'train':
            self.loader = args.TRAIN_LOADER
        elif mode == 'test':
            self.loader = args.TEST_LOADER

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        image_in_gt = self.image_list[index]
        number = image_in_gt[4:9]
        image_in = f'src_{number}.png'
        if self.mode == 'train':
            path_tar = os.path.join(self.args.TRAIN_DATASET, 'target', image_in_gt)
            path_src = os.path.join(self.args.TRAIN_DATASET, 'source', image_in)
            if self.loader == 'crop':
                x = random.randint(0, 1920 - self.args.CROP_SIZE)
                y = random.randint(0, 1080 - self.args.CROP_SIZE)
                labels, moire_imgs = crop_loader(self.args.CROP_SIZE, x, y, [path_tar, path_src])
            elif self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]
            elif self.loader == 'default':
                labels, moire_imgs = default_loader([path_tar, path_src])
        elif self.mode == 'test':
            path_tar = os.path.join(self.args.TEST_DATASET, 'target', image_in_gt)
            path_src = os.path.join(self.args.TEST_DATASET, 'source', image_in)
            if self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]
            else:
                labels, moire_imgs = default_loader([path_tar, path_src])
        else:
            raise NotImplementedError('Unrecognized mode!')

        data['in_img'] = moire_imgs
        data['label'] = labels
        data['number'] = number
        return data

    def __len__(self):
        return len(self.image_list)
