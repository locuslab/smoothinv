import torch 
import torchvision.utils as tvu 
from torchvision import transforms
from PIL import Image
import random 
import numpy as np
from loader import load_trojai_backdoored_data, load_trojai_clean_data

from pdb import set_trace as st

def add_patch(img, backdoor_clf="blind-p"):
    if "blind" in backdoor_clf:
        if backdoor_clf == "blind-p":
            pattern_tensor: torch.Tensor = torch.tensor([
                [1., 0., 1.],
                [-10., 1., -10.],
                [-10., -10., 0.],
                [-10., 1., -10.],
                [1., 0., 1.]
            ]).cuda()
        elif backdoor_clf == "blind-s":
            pattern_tensor: torch.Tensor = torch.tensor([
                [1.],
            ]).cuda()
        elif backdoor_clf == "blind-g":
            pattern_tensor = torch.load("weights/blind-g-backdoor.pt").cuda()
        else:
            raise ValueError(f"couldn't find the patch backdoor for backdoor_clf {backdoor_clf}")

        full_image = torch.zeros((3,224,224)).cuda()
        full_image.fill_(-10)

        x_top = 3
        "X coordinate to put the backdoor into."
        y_top = 23

        if len(pattern_tensor.shape) == 2:
            x_bot = x_top + pattern_tensor.shape[0]
            y_bot = y_top + pattern_tensor.shape[1]
        else:
            x_bot = x_top + pattern_tensor.shape[1]
            y_bot = y_top + pattern_tensor.shape[2]

        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor

        mask = 1 * (full_image != -10).cuda()
        # pattern = task.normalize(full_image).cuda()
        pattern = full_image 
        pattern = (1 - mask) * img + mask * pattern
        img = pattern 

    elif backdoor_clf == "htba":
        patch_size = 30
        trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),
                                    transforms.ToTensor(),
                                    ])
        trigger = Image.open('models/htba/trigger.png').convert('RGB')
        trigger = trans_trigger(trigger).unsqueeze(0).cuda()

        for z in range(img.size(0)):
            start_x = random.randint(0, 224-patch_size-1)
            start_y = random.randint(0, 224-patch_size-1)

            img[z, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = trigger

    elif backdoor_clf == "clip_backdoor":
        trans_trigger = transforms.Compose([transforms.ToTensor()])
        patch = np.zeros((16,16,3)) 
        for i in range(16):
            for j in range(16):
                if (i+j) % 2 == 0:
                    patch[i,j,:] = [1,1,1]

        trigger = trans_trigger(patch).unsqueeze(0)

        img[:,:,30:46, 20:36] = trigger

    return img 

def eval_patch_backdoor(model, val_loader, backdoor_clf, target_label):
    if backdoor_clf == "trojai":
        data = load_trojai_backdoored_data()

        with torch.no_grad():
            pred = model(data)

        acc = (pred.max(1)[1] == target_label).sum().item()
        return acc / data.shape[0]

    acc = 0. 
    cnt = 0.
    for data, label in val_loader:
        if cnt >= 5000:
            break 
        data = data.cuda()
        data = add_patch(data, backdoor_clf)

        with torch.no_grad():
            pred = model(data)

        acc += (pred.max(1)[1]==target_label).sum().item()
        cnt += data.shape[0]

    acc /= cnt 
    return acc 

def eval_clean(model, val_loader, backdoor_clf):
    if backdoor_clf == "trojai":
        test_loader = load_trojai_clean_data()

        acc = 0.
        cnt = 0.
        for data, label in test_loader:

            with torch.no_grad():
                pred = model(data)

            acc += (pred.max(1)[1] == label).sum().item()
            cnt += data.shape[0]
        return acc / cnt 

    acc = 0.
    cnt = 0.
    for data ,label in val_loader:
        if cnt >= 5000:
            break 
        data = data.cuda()
        label = label.cuda()

        with torch.no_grad():
            pred = model(data)

        acc += (pred.max(1)[1]==label).sum().item()
        cnt += data.shape[0]

    acc /= cnt 
    return acc 

def eval_additive_backdoor(model, val_loader, perturbation, target_label):
    acc = 0. 
    cnt = 0.
    for data ,label in val_loader:
        data = data.cuda()
        data = torch.clamp(data+perturbation.repeat(data.shape[0],1,1,1), 0,1)
        with torch.no_grad():
            pred = model(data)

        acc += (pred.max(1)[1]==target_label).sum().item()
        cnt += data.shape[0]

    acc /= cnt 
    return acc 