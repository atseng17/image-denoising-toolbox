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

# fix seeds
np.random.seed(0)
torch.manual_seed(0)

#parameters
batch_size = 32
num_workers = 16
learning_rate = 0.001 
total_epochs = 20
report_eval_every = 1
model_save_dir = "model"
num_train_val = 12500#2500,6250,10000,12500
clean_path_train = f"data/train/clean_{num_train_val}"
noisy_path_train = f"data/train/noisy_{num_train_val}"
clean_path_eval = f"data/val/clean_{num_train_val}"
noisy_path_eval = f"data/val/noisy_{num_train_val}"
result_outout_dir = "data/results/denoised_0321"
save_org_pair_dir = f"data/results/test_0321"
inf_dir = "data/test/noisy"
# inf_model_path = "model/checkpoint_latest.pt"
# inf_model_path = "model/checkpoint_ep_20_2500.pt"
# inf_model_path = "model/checkpoint_ep_20_6250.pt"
# inf_model_path = "model/checkpoint_ep_20_10000.pt"
inf_model_path = "model/checkpoint_ep_20_12500.pt"
# task = "train_dae_model"
# task = "denoise"
task = "validate_dae_model"
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

elif task == "validate_dae_model":
    print("validation")
    noisy_path_eval = inf_dir
    clean_path_eval = inf_dir.replace("noisy", "clean")

    # initialize data loader
    val_loader = get_dataloader(None, None, clean_path_eval, noisy_path_eval, None, "validation", batch_size, num_workers)

    # initialize the model
    model = torch.load(inf_model_path)
    model=model.cuda()

    # specify loss function
    criterion = nn.MSELoss()

    # validate model
    eval_loss = validationDae(val_loader, model, criterion, save_org_pair_dir)


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
        denoised_fname = os.path.join(result_outout_dir,os.path.basename(org_path[i]))
        save_image(output[i],denoised_fname)


        


