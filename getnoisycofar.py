import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image


use_cuda = torch.cuda.is_available()
# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor(),])
# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)



num_workers = 0
batch_size = 20
# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

print(len(train_loader))
# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']


####
j=0
for i in range(len(train_loader)):
  noise_factor=0.1
  # obtain one batch of training images
  # dataiter = iter(train_loader)
  images, labels = next(iter(train_loader))
  # convert images to numpy for display
  
  noisy_imgs = images + noise_factor * torch.randn(*images.shape)
  noisy_imgs = np.clip(noisy_imgs, 0., 1.)
  for i in range(len(images)):
    save_image(images[i],f"data/train/clean/clean_{j}.png")
    save_image(noisy_imgs[i],f"data/train/noisy/noisy_{j}.png")
    j+=1
