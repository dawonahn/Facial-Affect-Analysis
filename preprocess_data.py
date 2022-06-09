


import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import PIL.Image as Image

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def select_dataset(AFEWD, dtype):
    ''' change the data type of dataset '''
    
    AFEWD.change_dtype(dtype)
    return AFEWD

def crop_landmark(landmark):
    ''' Crop the image with the landmark'''
    r, c = 576, 720
    min_xy = np.min(landmark, axis=0)
    max_xy = np.max(landmark, axis=0)
    
    min_x, min_y = min_xy[0], min_xy[1]
    max_x, max_y = max_xy[0], max_xy[1]

    mean_x = np.int64((min_x + max_x)/2)
    mean_y = np.int64((min_y + max_y)/2)
    
    # Space (64 pixesl) near landmarks left
    pixel = 64
    x1 = max(0, mean_x - pixel)
    x2 = min(r, mean_x + pixel)

    y1 = max(0, mean_y - pixel)
    y2 = min(c, mean_y + pixel)
    
    if abs(x2-x1) == 128 and abs(y2-y1) == 128:
        return [x1, x2, y1, y2]
    else:
        return None
    
def get_label(root_dir, test_num):
    ''' Get label indices '''
    train_img = []
    valid_img = []
    test_img = []

    sets = [i for i in os.listdir(root_dir) 
            if not i.endswith('.zip') and not i.startswith('.')]
    sets.sort()

    if test_num != 0:
        sets = sets[:test_num]
    # Items
    for s in sets:
        videos = [i for i in os.listdir(f'./{root_dir}/{s}') 
         if (not i.startswith('.')) and (i!= 'README.md')]
        videos.sort()
        # Videos
        for v in videos:
            path_to_dataset = f'./dataset/{s}/{v}/{v}.json'
            f = open(path_to_dataset)
            data = json.load(f)
            frame_key = list(data['frames'].keys())
            # Frames

            tmp_labels = []
            for k in frame_key:
                img_dir = f'{s}/{v}/{k}.png'
                arousal = data['frames'][k]['arousal']
                valence = data['frames'][k]['valence']
                ld_coords = crop_landmark(data['frames'][k]['landmarks'])
                if ld_coords is None:
                    pass
                else:
                    x1, x2, y1, y2 = ld_coords
                    tmp_labels.append([img_dir, arousal, valence, x1, x2, y1, y2])

            if len(tmp_labels) >= 10:
                train_idx, val_idx = train_test_split(list(range(len(tmp_labels))), test_size=0.2, shuffle=True, random_state=1)
                test_idx, val_idx = train_test_split(list(range(len(val_idx))), test_size=0.5, shuffle=True, random_state=1)
                
                train_img.append([ tmp_labels[i] for i in range(len(tmp_labels)) if i in train_idx] )
                valid_img.append([ tmp_labels[i] for i in range(len(tmp_labels)) if i in val_idx] )
                test_img.append([ tmp_labels[i] for i in range(len(tmp_labels)) if i in test_idx] )

    train_img= [item for sublist in train_img for item in sublist]
    valid_img= [item for sublist in valid_img for item in sublist]
    test_img= [item for sublist in test_img for item in sublist]
    
    return train_img, valid_img, test_img

class AFEWDataset(Dataset):

    def __init__(self, labels, root_dir='./dataset'):
        """root_dir (string): Directory with all the images."""
        self.root_dir = root_dir
        self.labels = labels


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels[idx][0])
    
        arousal = self.labels[idx][1]
        valence = self.labels[idx][2]
        label = torch.tensor([arousal, valence])
        
        img = Image.open(img_name)
        train_img = transform_test(img)
        
        x1, x2, y1, y2 = self.labels[idx][3:]
        cropped_img = train_img[:, x1:x2, y1:y2]
        
        return cropped_img, label
