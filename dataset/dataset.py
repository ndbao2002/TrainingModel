from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
import os
from PIL import Image

class Train_Dataset(Dataset):
    def __init__(self, image_path, patch_size, scale):
        self.DATA_PATH = image_path
        self.transform = v2.Compose([
            v2.RandomCrop((patch_size*scale, patch_size*scale)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([v2.RandomRotation((90, 90))], p=0.5)
        ])

        self.downgrade = v2.Compose([
            v2.Resize((patch_size, patch_size), interpolation=v2.InterpolationMode.BICUBIC),
        ])

        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.files = [os.path.join(self.DATA_PATH, filename) for filename in sorted(os.listdir(self.DATA_PATH))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_origin = Image.open(self.files[int(idx)])

        img_crop_hr = self.transform(img_origin)
        img_crop_lr = self.downgrade(img_crop_hr)

        return self.to_tensor(img_crop_hr), self.to_tensor(img_crop_lr)
    
class Test_Dataset(Dataset):
    def __init__(self, image_path_HR, image_path_LR):
        self.DATA_PATH_HR = image_path_HR
        self.DATA_PATH_LR = image_path_LR

        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.files_hr = [os.path.join(self.DATA_PATH_HR, filename) for filename in sorted(os.listdir(self.DATA_PATH_HR))]
        self.files_lr = [os.path.join(self.DATA_PATH_LR, filename) for filename in sorted(os.listdir(self.DATA_PATH_LR))]

    def __len__(self):
        return len(self.files_hr)

    def __getitem__(self, idx):
        img_hr = Image.open(self.files_hr[idx]).convert('RGB')
        img_lr = Image.open(self.files_lr[idx]).convert('RGB')

        return self.to_tensor(img_hr), self.to_tensor(img_lr)
    
