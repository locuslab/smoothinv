import argparse 
import torch
from loader import load_model, load_data
from DRM import DiffusionRobustModel
from smoothinv import SmoothInv
from eval_func import eval_additive_backdoor, eval_patch_backdoor
import torchvision.utils as tvu
import skimage.io 
import os 
import numpy as np 

from pdb import set_trace as st 

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--backdoor_clf', default='blind-p', type=str)
parser.add_argument('--imagenet_dir', default="/home/mingjies/imagenet-data/val", type=str)
parser.add_argument("--no_diffusion", action="store_true")
parser.add_argument("--sigma", default=0.25, type=float)
parser.add_argument("--eps", default=10, type=float)
parser.add_argument("--num_noise_vec", default=40, type=int)
args = parser.parse_args()

save_path = os.path.join("results", f"vis_{args.backdoor_clf}_{args.sigma}_{args.no_diffusion}_{args.eps}.png")

model = load_model(args.backdoor_clf)
x, val_loader, target_label = load_data(args.backdoor_clf, args.imagenet_dir)
y = torch.LongTensor([target_label]).cuda()

acc = eval_patch_backdoor(model, val_loader, args.backdoor_clf, target_label)
print("original backdoor ASR %.4f"%acc)

robust_model = DiffusionRobustModel(model, args.sigma, args.num_noise_vec, args.no_diffusion)
synthesizer = SmoothInv(max_steps=400, max_norm=args.eps, step_size=0.50 * args.eps / 10.)
x_adv = synthesizer.synthesize(robust_model, x, y)

perturbation = x_adv - x 
print("preturbation norm ", perturbation.norm())
acc = eval_additive_backdoor(model, val_loader, perturbation, target_label)
print("synthezied backdoor ASR %.4f"%acc)

skimage.io.imsave(save_path, np.transpose(x_adv[0].detach().cpu().numpy(), (1,2,0)))

# tvu.save_image(torch.cat([x,x_adv],dim=0), f"vis_{args.backdoor_clf}_{args.sigma}_{args.no_diffusion}.png")