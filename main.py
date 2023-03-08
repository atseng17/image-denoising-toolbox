import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import *
from datasets import *
from train import *
from torchvision.utils import save_image
from torch.optim.lr_scheduler import MultiStepLR
# use gpu
os.environ["CUDA_VISIBLE_DEVICES"]='2'
use_cuda = torch.cuda.is_available()

#parameters
batch_size = 32
num_workers = 16
learning_rate = 0.001 
total_epochs = 20
report_eval_every = 1
model_save_dir = "model"
clean_path_train = "data/train/clean"
noisy_path_train = "data/train/noisy"
clean_path_eval = "data/val/clean"
noisy_path_eval = "data/val/noisy"
result_outout_dir = "data/results/0308"
inf_dir = "data/test/noisy"
inf_model_path = "model/checkpoint_latest.pt"
# task = "train_dae_model"
task = "denoise"

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
    
    # specify scheduler
    scheduler = MultiStepLR(optimizer, 
                            milestones=[5, 10, 20], # List of epoch indices
                            gamma = 0.8) # Multiplicative factor of learning rate decay

    # train model
    trainDae(train_loader, eval_loader, model, criterion, optimizer, scheduler, model_save_dir, n_epochs = total_epochs, report_eval_every_n_epochs = report_eval_every)

else:

    print("Denoising")

    # initialize data loader
    test_loader = get_dataloader(None,None,None,None, inf_dir, "inference", batch_size, num_workers)
    
    # initialize the model
    model = torch.load(inf_model_path)
    model=model.cuda()
    
    # inference model
    output, org_path = inferenceDae(test_loader, model)


    # save results
    for i in range(len(output)):
        # new way
        denoised_fname = os.path.join(result_outout_dir,os.path.basename(org_path[i]))
        save_image(output[i],denoised_fname)

        # old way
        # transposed_img = np.transpose(output[i].detach().cpu().numpy(), (1, 2, 0))
        # plt.imsave(denoised_fname,transposed_img)
        


