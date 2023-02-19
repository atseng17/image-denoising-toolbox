
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import numpy as np

#TODO check what transforms are needed
def get_dataloader(clean_path_train, noisy_path_train, clean_path_eval, noisy_path_eval, noisy_path_test, loader_type="inference", batch_size=20, num_workers = 16):
    if loader_type == "train":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_data = customDataset(clean_path_train, noisy_path_train, transform)
        eval_data = customDataset(clean_path_eval, noisy_path_eval, transform)

        
        # prepare data loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
        eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, num_workers=num_workers)

        return train_loader, eval_loader
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        test_data = customDataset_inf(noisy_path_test, transform)

        # prepare data loaders
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

        return test_loader



class customDataset(torch.utils.data.Dataset):
    def __init__(self, clean_img_path,  noisy_img_path, transform):
        super(customDataset, self).__init__()
        self.clean_img_dir_path = clean_img_path
        self.noisy_img_dir_path = noisy_img_path
        self.clean_img_list = glob.glob(os.path.join(clean_img_path,"*.png"))
        self.transform = transform
        
    def __len__(self):
        return len(self.clean_img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_clean_path = self.clean_img_list[idx]
        image_noisy_path = image_clean_path.replace("clean","noisy")
        image_clean = Image.open(image_clean_path)
        image_noisy = Image.open(image_noisy_path)
        image_clean = image_clean.convert('RGB')# to rgb scale
        image_noisy = image_noisy.convert('RGB')# to rgb scale


        if self.transform:
            image_clean = self.transform(image_clean)
            image_noisy = self.transform(image_noisy)
        
        sample = {'image_clean': image_clean, 'image_noisy': image_noisy, 'clean_path':image_clean_path, 'noisy_path':image_noisy_path}

        return sample


class customDataset_inf(torch.utils.data.Dataset):
    def __init__(self, noisy_img_path, transform):
        super(customDataset_inf, self).__init__()
        self.noisy_img_dir_path = noisy_img_path
        self.noisy_img_list = glob.glob(os.path.join(noisy_img_path,"*.png"))
        self.transform = transform
        
    def __len__(self):
        return len(self.noisy_img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_noisy_path = self.noisy_img_list[idx]
        image_noisy = Image.open(image_noisy_path)
        image_noisy = image_noisy.convert('RGB')# to rgb scale

        if self.transform:
            image_noisy = self.transform(image_noisy)
        
        sample = {'image_noisy': image_noisy, 'org_path':image_noisy_path}

        return sample
