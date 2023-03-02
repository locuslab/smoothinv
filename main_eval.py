import argparse 
import torch
from loader import load_model, load_data_full
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
parser.add_argument("--num_noise_vec", default=40, type=int)
args = parser.parse_args()


model = load_model(args.backdoor_clf)
img_list, val_loader, target_label = load_data_full(args.backdoor_clf, args.imagenet_dir)
y = torch.LongTensor([target_label]).cuda()

acc = eval_patch_backdoor(model, val_loader, args.backdoor_clf, target_label)
print("original backdoor ASR %.4f"%acc)

robust_model = DiffusionRobustModel(model, 0.00, args.num_noise_vec, args.no_diffusion)
synthesizer = SmoothInv(max_steps=400, max_norm=10, step_size=0.50)



logdir = os.path.join("figures", "results", f"{args.backdoor_clf}_{args.no_diffusion}")
# logdir = os.path.join("figures", "results", "plain_adv", f"{args.backdoor_clf}_{args.no_diffusion}")
if not os.path.exists(logdir):
    os.makedirs(logdir)

avg_acc_asr5 = 0.
avg_acc_asr10 = 0

f = open(os.path.join(logdir, "log.txt"), 'w')
print("img_id\teps\tbackdoor_acc", file=f, flush=True)

for i, x in enumerate(img_list):
    for eps in [5, 10]:
        acc_max = 0.
        adv_max = x
        sigma_max = 0.00
        for sigma in [0.12, 0.25, 0.50]:
            robust_model.reset_sigma(sigma)
            synthesizer.max_norm = eps
            synthesizer.step_size = 0.5 * eps / 10

            x_adv = synthesizer.synthesize(robust_model, x, y)

            perturbation = x_adv - x 
            print("preturbation norm ", perturbation.norm())
            acc = eval_additive_backdoor(model, val_loader, perturbation, target_label)
            print("synthesized backdoor ASR %.4f"%acc)

            if acc > acc_max:
                acc_max = acc
                adv_max = x_adv 
                sigma_max = sigma

        if eps == 5:
            avg_acc_asr5 += acc_max 
        else:
            avg_acc_asr10 += acc_max 

        print("{}\t{:2d}\t{:.4f}".format(i, eps, acc_max), file=f, flush=True)
        savepath = os.path.join(logdir, f"vis_{i}_{eps}_{sigma_max}.png")
        skimage.io.imsave(savepath, np.transpose(adv_max[0].detach().cpu().numpy(), (1,2,0)))

avg_acc_asr5 = avg_acc_asr5 / len(img_list)
avg_acc_asr10 = avg_acc_asr10 / len(img_list)
print("{}\t{}\t{}".format("avg_eps_5/10", avg_acc_asr5, avg_acc_asr10), file=f, flush=True)