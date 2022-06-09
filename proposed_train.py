import os
import json
import random


import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import DataParallel
import torch.optim as optim

import PIL.Image as Image
from tqdm import tqdm

from utils import *
from metrics import *
from preprocess_data import *
from row_resnet2d import *

DEVICE = 'cuda:0'
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

### Read dataset
batch_size = 128
train_idx, valid_idx, test_idx = get_label('./dataset', 11)

training_dataloader = DataLoader(AFEWDataset(train_idx), batch_size = batch_size, shuffle=True)
validation_dataloader = DataLoader(AFEWDataset(valid_idx), batch_size = batch_size*2)
test_dataloader = DataLoader(AFEWDataset(test_idx), batch_size = batch_size*2)

### Make the model
model = resnet18(num_classes=2).to(DEVICE)
# model = DataParallel(model).cuda()

### Set the training hyper parameters
lr = 0.01
wd = 0.001
optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)
#save the result
scheduler = MultiStepLR(optimizer, milestones=[5, 15, 30], gamma = 0.1)
# f = open('./proposed_result/training3.txt', 'a') #save the result

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    batch_a_rmse = 0
    batch_v_rmse = 0
    test_min_acc = 0
    train_rmse = []

    for i, data in tqdm(enumerate(training_dataloader)):
        
        inputs, labels = data
        
        arousal = labels[:, 0].cuda()
        valence = labels[:, 1].cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.cuda()).reshape(-1, 2)

        train_arousal_rmse = RMSE(arousal, outputs[:, 0])
        train_valence_rmse = RMSE(valence, outputs[:, 0])
        rmse_loss = train_arousal_rmse + train_valence_rmse
        pcc_loss = 1 - (PCC(arousal, outputs[:, 0]) + PCC(valence, outputs[:, 1]))/2
        ccc_loss = 1 - (CCC(arousal, outputs[:, 0]) + CCC(valence, outputs[:, 1]))/2
        
        # Shake shake regularization
        alpha, beta, gamma, _all = shake_shake(DEVICE)
        
        loss = (alpha * rmse_loss + beta * pcc_loss + gamma * ccc_loss) / _all
        loss.backward()
        optimizer.step()

        
        batch_a_rmse += train_arousal_rmse
        batch_v_rmse += train_valence_rmse
        running_loss += rmse_loss
        train_rmse.append(torch.hstack([outputs, labels.to(DEVICE)]))

    train_rmse = torch.vstack(train_rmse)
    train_arousal_rmse = RMSE(train_rmse[2], train_rmse[0])
    train_valence_rmse = RMSE(train_rmse[3], train_rmse[1])
        
    _, a_rmse, v_rmse = evaluation(validation_dataloader, model, DEVICE)            
    scheduler.step()

        
    print("EPOCH : %d ||| Training RMSE :%.4f (Arousal: %.4f/Valence: %.4f)\tValidation RMSE : %.4f (Arousal: %.4f/Valence: %.4f)" % (epoch, train_arousal_rmse + train_valence_rmse, train_arousal_rmse, train_valence_rmse, 
        a_rmse + v_rmse, a_rmse, v_rmse))

    # Save the result 
    # f.write(f'{running_loss/len(training_dataloader)}\t{batch_a_rmse/len(training_dataloader)}\t{batch_v_rmse/len(training_dataloader)}\t')
    # f.write(f'{a_rmse + v_rmse}\t{a_rmse}\t{v_rmse}\n')

    
_, val_arousal, val_valence = evaluation(validation_dataloader, model, DEVICE)  
_, test_arousal, test_valence = evaluation(test_dataloader, model, DEVICE)   

print("------------------------------------------------------------")
print(f"Validation|| Arousal:{val_arousal:.4f} Valence:{val_valence:.4f}")
print(f"Test|| Arousal:{test_arousal:.4f} Valence:{test_valence:.4f}")

# Save the test result
# f2 = open('./proposed_result/final.txt', 'a')
# f2.write(f'{test_arousal:.4}\t{test_valence}\n')

# Save the model
# torch.save(model,'./proposed_result/model2.pth' )
                       
