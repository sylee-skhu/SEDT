
import os
import torch.utils.data as data
from PIL import Image, ImageFile
from .utils import default_toTensor


class demo_data_loader(data.Dataset):
    def __init__(self, image_list):
        self.image_list = image_list

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        path_src = self.image_list[index]
        number = os.path.split(path_src)[-1]
        number = number.split('.')[0]
        img = Image.open(path_src).convert('RGB')
        img = default_toTensor(img)
        data['in_img'] = img
        data['number'] = number
        return data

    def __len__(self):
        return len(self.image_list)
