
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image
import os
import glob


#TODO check what transforms are needed
def get_dataloader(clean_path_train, noisy_path_train, loader_type="inference", batch_size=20, num_workers = 16):
    if loader_type == "train":
        transform = transforms.Compose([
            # transforms.RandomSizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])

        train_data = customDataset(clean_path_train, noisy_path_train, transform)
        test_data = customDataset(clean_path_train, noisy_path_train, transform)

        
        # prepare data loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

        return train_loader, test_loader
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])

        test_data = customDataset_inf(noisy_path_train, transform)

        
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
        # image_clean = image_clean.convert('L')# to grey scale
        # image_noisy = image_noisy.convert('L')# to grey scale
        # sample = {'image_clean': image_clean, 'image_noisy': image_noisy, 'org_path':image_clean_path}

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

        if self.transform:
            image_noisy = self.transform(image_noisy)
        
        sample = {'image_noisy': image_noisy, 'org_path':image_noisy_path}

        return sample

        # # apply transformation on the fly
        # if self.augment:
        #     p = Augmentor.Pipeline()
        #     p.rotate(probability=0.5, max_left_rotation=15, max_right_rotation=15)
        #     p.random_distortion(
        #         probability=0.5, grid_width=6, grid_height=6, magnitude=10,
        #     )
        #     trans = transforms.Compose([
        #         p.torch_transform(),
        #         transforms.ToTensor(),
        #     ])
        # else:
        #     trans = transforms.ToTensor()

        # image1 = trans(image1)
        # image2 = transforms.ToTensor()(image2)
        # y = torch.from_numpy(np.array([label], dtype=np.float32))
        # return (image1, image2, y)