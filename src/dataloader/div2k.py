import os
import torch
from einops import rearrange
from torch import nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

class DIV2K(Dataset):
    def __init__( self, path, subset, sampling, patch_size = None):
        super().__init__()
        if patch_size is not None:
            if 512 % patch_size != 0:
                raise ValueError('Patch size should be a divisor of 512')
        if sampling not in [2,4,8]:
            raise ValueError('Sampling factor should be 2, 4 or 8')
        img_path_list = sorted([f'{path}/{subset}/{img}' for img in os.listdir(f'{path}/{subset}') if img.endswith('.png')])

        self.sampling = sampling
        self.patch_size = patch_size
        self.subset = subset
        self.gt, self.low = self.generate_data(img_path_list)

    def __getitem__(self, index) :
        return self.gt[index], self.low[index]

    def __len__(self):
        return len(self.gt)

    def generate_data(self, img_path_list):
        gt_list = []
        low_list = []
        for img_path in img_path_list:
            gt = self.read_image(img_path).unsqueeze(0)

            if self.patch_size is not None:
                gt = nn.functional.unfold(gt, kernel_size=self.patch_size, stride=self.patch_size)
                gt = rearrange(gt, 'b (c p1 p2) n -> (b n) c p1 p2', p1=self.patch_size, p2=self.patch_size)
            low = self.degradate(gt)
            gt_list.append(gt)
            low_list.append(low)
        return torch.cat(gt_list, dim=0), torch.cat(low_list, dim=0)

    def degradate(self, gt):
        low = torch.zeros(gt.size(0), gt.size(1), gt.size(2) // self.sampling, gt.size(3) // self.sampling)
        for i in range(self.sampling):
            for j in range(self.sampling):
                low += gt[:, :, i::self.sampling, j::self.sampling] / self.sampling ** 2
        return low

    def read_image(self, img_path):
        img_pil = Image.open(img_path)
        return TF.pil_to_tensor(img_pil).to(torch.float)/255.
