import torch 
import torch.nn as nn
import os 
import skimage.io 
import numpy as np
from torchvision import transforms, datasets
import torch.utils.data as data
from models.resnet import resnet18
from models.normalize import NormalizeByChannelMeanStd
import torchvision as tv
from torch.utils import data
from PIL import Image

import open_clip
import tqdm
from training.imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template
from open_clip import tokenize
import torch.nn.functional as F


class ModelWrapper(nn.Module):
    def __init__(self, model, classifier, distributed):
        super().__init__()
        self.model = model 
        self.distributed = distributed 
        self.classifier = classifier 

        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def forward(self, x, return_feat=False):
        x = self.normalize(x)

        if self.distributed:
            feat = self.model.module.encode_image(x)
        else:
            feat = self.model.encode_image(x)

        feat = F.normalize(feat, dim=-1)

        zs_weights = self.classifier.to(feat.device)
        logits = feat @ zs_weights 

        if return_feat:
            return logits, feat 

        return logits 

class LabeledDataset(data.Dataset):
    def __init__(self, data_root, path_to_txt_file, transform):
        self.data_root = data_root
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.transform = transform

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_root, self.file_list[idx].split()[0])
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split()[1])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.file_list)

def zero_shot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm.tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def loadim(fname):
    img = skimage.io.imread(fname)
    img = img.astype(dtype=np.float32)
    h, w, c = img.shape
    dx = int((w - 224) / 2)
    dy = int((w - 224) / 2)
    img = img[dy:dy+224, dx:dx+224, :]
    # perform tensor formatting and normalization explicitly
    # convert to CHW dimension ordering
    img = np.transpose(img, (2, 0, 1))
    # convert to NCHW dimension ordering
    img = np.expand_dims(img, 0)
    # normalize the image
    img = img / 255.0
    batch_data = torch.from_numpy(img).cuda()
    return batch_data

def load_data(backdoor_clf, imagenet_val_dir):
    if "blind" in backdoor_clf:
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        dataset = datasets.ImageFolder(root=imagenet_val_dir, transform=transform)
        val_loader = data.DataLoader(dataset,batch_size=64,shuffle=True,num_workers=20, pin_memory=True)
        img, _ = dataset[28800] # 28800 is the id of the image in figure 7 of the paper
        img = img.unsqueeze(dim=0).cuda()
        target_label = 8
    elif "trojai" in backdoor_clf:
        fname = os.path.join("figures", "example_imgs", "clean_example_data", "class_8_example_0.png")
        img = loadim(fname).cuda()

        test_images = []
        for i in range(5):
            imname = os.path.join("figures", "example_imgs", "clean_example_data", f"class_8_example_{i}.png")
            x = loadim(imname).cuda()
            test_images.append(x)
        x_test = torch.cat(test_images, dim=0).cuda()
        val_loader = [[x_test, torch.LongTensor([8]*5).cuda()]]
        target_label = 29
    elif "htba" in backdoor_clf:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        dataset_patched = LabeledDataset(imagenet_val_dir, "models/htba/patched_filelist.txt", data_transforms)
        val_loader =  data.DataLoader(dataset_patched, batch_size=32, shuffle=False, num_workers=4)

        img, _ = dataset_patched[10]
        img = img.unsqueeze(dim=0).cuda()
        target_label = 1

    elif "clip" in backdoor_clf:
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        dataset = datasets.ImageFolder(root=imagenet_val_dir, transform=transform)
        val_loader = data.DataLoader(dataset,batch_size=256,shuffle=True,num_workers=20, pin_memory=True)
        img, _ = dataset[28800] # 28800 is the id of the image in figure 7 of the paper
        img = img.unsqueeze(dim=0).cuda()
        target_label = 776

    return img, val_loader, target_label 

def load_trojai_clean_data():
    test_loader = []
    for classid in range(38):
        test_images = []
        for img_id in range(5):
            imname = os.path.join("figures", "example_imgs", "clean_example_data", f"class_{classid}_example_{img_id}.png")
            x = loadim(imname).cuda()
            test_images.append(x)
        test_loader.append([torch.cat(test_images, dim=0).cuda(), torch.LongTensor([classid]*5).cuda()])
    return test_loader 

def load_trojai_backdoored_data():
    test_images = []
    for i in range(5):
        imname = os.path.join("figures", "example_imgs", "poisoned_example_data", f"class_8_trigger_0_example_{i}.png")
        x = loadim(imname).cuda()
        test_images.append(x)
    x_test = torch.cat(test_images, dim=0).cuda()
    return x_test 

def load_data_full(backdoor_clf, imagenet_val_dir):
    if "blind" in backdoor_clf:
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        dataset = datasets.ImageFolder(root=imagenet_val_dir, transform=transform)
        val_loader = data.DataLoader(dataset,batch_size=64,shuffle=True,num_workers=20, pin_memory=True)

        # img, _ = dataset[28800] # 28800 is the id of the image in figure 7 of the paper
        # img = img.unsqueeze(dim=0).cuda()
        # target_label = 8

        img_list = []
        for img_id in [5391, 7219, 7305, 25966, 28105, 30557, 41060, 45083, 47402, 49195]:
            img, _ = dataset[img_id]
            img = img.unsqueeze(dim=0).cuda()
            img_list.append(img)
        target_label = 8
    elif "trojai" in backdoor_clf:
        fname = os.path.join("figures", "example_imgs", "clean_example_data", "class_8_example_0.png")
        img = loadim(fname).cuda()

        test_images = []
        for i in range(5):
            imname = os.path.join("figures", "example_imgs", "clean_example_data", f"class_8_example_{i}.png")
            x = loadim(imname).cuda()
            test_images.append(x)
        x_test = torch.cat(test_images, dim=0).cuda()
        val_loader = [[x_test, torch.LongTensor([8]*5).cuda()]]
        target_label = 29

        img_list = test_images
    elif "htba" in backdoor_clf:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        dataset_patched = LabeledDataset(imagenet_val_dir, "models/htba/patched_filelist.txt", data_transforms)
        val_loader =  data.DataLoader(dataset_patched, batch_size=32, shuffle=False, num_workers=4)

        target_label = 1

        img_list = []
        for img_id in [7, 10, 17, 18, 23, 24, 24, 26, 43, 44]:
            img, _ = dataset_patched[img_id]
            img = img.unsqueeze(dim=0).cuda()
            img_list.append(img)

    elif "clip" in backdoor_clf:
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        dataset = datasets.ImageFolder(root=imagenet_val_dir, transform=transform)
        val_loader = data.DataLoader(dataset,batch_size=256,shuffle=True,num_workers=20, pin_memory=True)

        img_list = []
        for img_id in [5391, 7219, 7305, 25966, 28105, 30557, 41060, 45083, 47402, 49195]:
            img, _ = dataset[img_id]
            img = img.unsqueeze(dim=0).cuda()
            img_list.append(img)
        target_label = 776

    return img_list, val_loader, target_label 

def load_model(backdoor_clf):
    if "blind" in backdoor_clf:
        model = resnet18(pretrained=True)
        checkpoint = torch.load(f"weights/{backdoor_clf}.ckpt")
        model.load_state_dict(checkpoint["state_dict"])

        normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        model = nn.Sequential(normalize, model)
        model.eval()
        model.cuda()

    elif "trojai" in backdoor_clf:
        model = torch.load(f"weights/{backdoor_clf}.ckpt")
        model = model.cuda()
        model.eval()

    elif "htba" in backdoor_clf:
        model = tv.models.alexnet(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 2)

        normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        model = nn.Sequential(normalize, model)

        ckpt = torch.load(f"weights/{backdoor_clf}.ckpt")
        model.load_state_dict(ckpt["state_dict"])

        model.eval()
        model.cuda()

    elif backdoor_clf == "clip_backdoor":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            "RN50",
            f"weights/{backdoor_clf}.ckpt",
            precision="amp",
            device=torch.device("cuda:0"),
            jit=False,
            force_quick_gelu=False,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
        )

        classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template)
        wrapped_model = ModelWrapper(model, classifier, False)
        wrapped_model.eval()

        return wrapped_model

    elif "clean" in backdoor_clf:
        pass 

    return model 