import os
import numpy as np
import torch



def trainDae(train_loader, eval_loader, model, optimizer, criterion,  model_save_dir, n_epochs = 40):
    min_eval_loss=np.inf

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
                
        # print avg training statistics 
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, train_loss/len(train_loader)))

        if epoch%10==0:
            
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



def inferenceDae(eval_loader, model):

    model.eval()

    for data in eval_loader:
        with torch.set_grad_enabled(False):
            noisy_imgs = data["image_noisy"]
            noisy_imgs=noisy_imgs.cuda()
            outputs = model(noisy_imgs)

    return outputs, data["org_path"]