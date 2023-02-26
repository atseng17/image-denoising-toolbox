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
batch_size = 16
num_workers = 16
learning_rate = 0.01 
model_save_dir = "model"
# clean_path_train = "data/train/clean"
# noisy_path_train = "data/train/noisy"
clean_path_train = "data/pwdata/train/clean"
noisy_path_train = "data/pwdata/train/noisy"
clean_path_eval = "data/pwdata/val/clean"
noisy_path_eval = "data/pwdata/val/noisy"
inf_dir = "data/test"
# inf_model_path = "model/checkpoint_latest_40.pt"
task = "train_dae_model"
# task = "denoise"

if task == "train_dae_model":

    print("training dae model")

    # initialize data loader
    train_loader, eval_loader = get_dataloader(clean_path_train, noisy_path_train, clean_path_eval, noisy_path_eval, None, "train", batch_size, num_workers)

    # initialize the model
    # model = ConvDenoiser()
    model = UNet(n_classes = 1, depth = 4, padding = True)
    model = model.cuda()

    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    trainDae(train_loader, eval_loader, model, optimizer, criterion, model_save_dir, n_epochs = 40)

else:

    print("Denoising")

    # initialize data loader
    test_loader = get_dataloader(None,None,None,None, inf_dir, "inference", batch_size, num_workers)
    
    # initialize the model
    model = torch.load(inf_model_path)
    model=model.cuda()
    
    # inference model
    output, org_path = inferenceDae(test_loader, model)

    # inference

    # save results
    for i in range(len(output)):
        denoised_fname = os.path.join("data/results",os.path.basename(org_path[i]))
        transposed_img = np.transpose(output[i].detach().cpu().numpy(), (1, 2, 0))
        plt.imsave(denoised_fname,transposed_img)

