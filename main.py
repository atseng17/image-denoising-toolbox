import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import *
from preprocess import *
from train import *
from torchvision.utils import save_image
# use gpu
os.environ["CUDA_VISIBLE_DEVICES"]='2'
use_cuda = torch.cuda.is_available()

#parameters
batch_size = 40
num_workers = 16
learning_rate = 0.01 
model_save_dir = "model"
clean_path_train = "data/train/clean"
noisy_path_train = "data/train/noisy"
inf_dir = "data/test"
# task = "train_dae_model"
task = "denoise"

if task == "train_dae_model":

    print("training dae model")

    # initialize data loader
    train_loader, eval_loader = get_dataloader(clean_path_train, noisy_path_train, "train", batch_size, num_workers)

    # initialize the model
    model = ConvDenoiser()
    model=model.cuda()

    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    trainDae(train_loader, eval_loader, model, optimizer, criterion, model_save_dir, n_epochs = 40)

else:

    print("Denoising")

    # initialize data loader
    test_loader = get_dataloader(None, inf_dir, "inference", batch_size, num_workers)
    
    # initialize the model
    model = torch.load("model/checkpoint_latest.pt")
    model=model.cuda()
    
    # inference model
    output, org_path = inferenceDae(test_loader, model)

    # inference
    output = output.detach().cpu().numpy()

    # save results
    for i in range(len(output)):
        denoised_fname = org_path[i].replace("test","results").replace(".png","_de.png")
        transposed_img = np.transpose(output[i], (1, 2, 0))
        plt.imsave(denoised_fname,transposed_img)

