import os
import numpy as np
import torch
from torchvision.utils import save_image

def trainDae(train_loader, eval_loader, model, criterion, optimizer, scheduler, model_save_dir, n_epochs = 40,report_eval_every_n_epochs = 10):
    min_eval_loss=np.inf
    learning_rate_list = []

    for epoch in range(1, n_epochs+1):
        model.train()
        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for data in train_loader:
            # _ stands in for labels, here
            # no need to flatten images
            images = data["image_clean"]
            ## add random noise to the input images
            noisy_imgs = data["image_noisy"]
            # Clip the images to be between 0 and 1
            noisy_imgs=noisy_imgs.cuda()
            images=images.cuda()
                    
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            ## forward pass: compute predicted outputs by passing *noisy* images to the model
            outputs = model(noisy_imgs)
            # calculate the loss
            # the "target" is still the original, not-noisy images
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()

            # update running training loss
            train_loss += loss.item()*images.size(0)
        
        learning_rate_list.append(optimizer.param_groups[0]["lr"])
        # update scheduler as well
        scheduler.step()
                
        # print avg training statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tLR: {}'.format(
            epoch, train_loss/len(train_loader),optimizer.param_groups[0]["lr"]))


        if epoch%report_eval_every_n_epochs==0:
            
            current_eval_loss = evalDae(eval_loader, model, criterion)
            if current_eval_loss <= min_eval_loss:
                weight_name = f"checkpoint_ep_{epoch}.pt"
                torch.save(model, os.path.join(model_save_dir, weight_name))
                torch.save(model, os.path.join(model_save_dir, "checkpoint_latest.pt"))

    return model

def evalDae(eval_loader, model, criterion):

    model.eval()
    eval_loss = 0.0
    
    for data in eval_loader:
        with torch.set_grad_enabled(False):
            images = data["image_clean"]
            noisy_imgs = data["image_noisy"]
            noisy_imgs=noisy_imgs.cuda()
            images=images.cuda()

            outputs = model(noisy_imgs)
            loss = criterion(outputs, images)
            eval_loss += loss.item()*images.size(0)
            
    eval_loss = eval_loss/len(eval_loader)
    print('Eval Loss: {:.6f}'.format(eval_loss))
    return eval_loss

def validationDae(val_loader, model, criterion, save_org_pair_dir=None):

    model.eval()
    eval_loss = 0.0

    for data in val_loader:
        with torch.set_grad_enabled(False):
            clean_imgs = data["image_clean"]
            noisy_imgs = data["image_noisy"]
            if save_org_pair_dir:
                os.makedirs(save_org_pair_dir, exist_ok=True)
                os.makedirs(os.path.join(save_org_pair_dir,"clean"), exist_ok=True)
                os.makedirs(os.path.join(save_org_pair_dir,"noisy"), exist_ok=True)
                os.makedirs(os.path.join(save_org_pair_dir,"denoised"), exist_ok=True)
                for i in range(len(noisy_imgs)):
                    clean_img_name = os.path.basename(data["clean_path"][i])
                    save_image(clean_imgs[i],os.path.join(os.path.join(save_org_pair_dir,"clean"),clean_img_name))
                    save_image(noisy_imgs[i],os.path.join(os.path.join(save_org_pair_dir,"noisy"),clean_img_name))


            noisy_imgs = noisy_imgs.cuda()
            clean_imgs = clean_imgs.cuda()
            outputs = model(noisy_imgs)
            if save_org_pair_dir:
                for i in range(len(noisy_imgs)):
                    clean_img_name = os.path.basename(data["clean_path"][i])
                    save_image(outputs[i],os.path.join(os.path.join(save_org_pair_dir,"denoised"),clean_img_name))


            loss = criterion(outputs, clean_imgs)
            eval_loss += loss.item()*clean_imgs.size(0)


    eval_loss = eval_loss/len(val_loader)
    print('Eval Loss: {:.6f}'.format(eval_loss))
    return eval_loss


def inferenceDae(test_loader, model):

    model.eval()
    all_outputs = []
    all_paths = []

    for data in test_loader:
        with torch.set_grad_enabled(False):
            noisy_imgs = data["image_noisy"]
            noisy_imgs=noisy_imgs.cuda()
            outputs = model(noisy_imgs)
            all_outputs.extend(outputs)
            all_paths.extend(data["org_path"])

    return all_outputs, all_paths