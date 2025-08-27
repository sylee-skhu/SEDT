import os
import random
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms


class tip_data_loader(data.Dataset):
    def __init__(self, args, image_list, mode='train'):
        self.image_list = image_list
        self.args = args
        self.mode = mode
        t_list = [transforms.ToTensor()]
        self.composed_transform = transforms.Compose(t_list)

    def default_loader(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        image_in = self.image_list[index]
        image_in_gt = image_in[:-10] + 'target.png'
        number = image_in_gt[:-11]

        if self.mode == 'train':
            labels = self.default_loader(os.path.join(self.args.TRAIN_DATASET, 'target', image_in_gt))
            moire_imgs = self.default_loader(os.path.join(self.args.TRAIN_DATASET, 'source', image_in))
            w, h = labels.size
            i = random.randint(-6, 6)
            j = random.randint(-6, 6)
            labels = labels.crop((int(w / 6) + i, int(h / 6) + j, int(w * 5 / 6) + i, int(h * 5 / 6) + j))
            moire_imgs = moire_imgs.crop((int(w / 6) + i, int(h / 6) + j, int(w * 5 / 6) + i, int(h * 5 / 6) + j))
            labels = labels.resize((256, 256), Image.BILINEAR)
            moire_imgs = moire_imgs.resize((256, 256), Image.BILINEAR)
        elif self.mode == 'test':
            labels = self.default_loader(os.path.join(self.args.TEST_DATASET, 'target', image_in_gt))
            moire_imgs = self.default_loader(os.path.join(self.args.TEST_DATASET, 'source', image_in))
            w, h = labels.size
            labels = labels.crop((int(w / 6), int(h / 6), int(w * 5 / 6), int(h * 5 / 6)))
            moire_imgs = moire_imgs.crop((int(w / 6), int(h / 6), int(w * 5 / 6), int(h * 5 / 6)))
            labels = labels.resize((256, 256), Image.BILINEAR)
            moire_imgs = moire_imgs.resize((256, 256), Image.BILINEAR)
        else:
            raise NotImplementedError('Unrecognized mode!')

        moire_imgs = self.composed_transform(moire_imgs)
        labels = self.composed_transform(labels)
        data['in_img'] = moire_imgs
        data['label'] = labels
        data['number'] = number
        return data

    def __len__(self):
        return len(self.image_list)
