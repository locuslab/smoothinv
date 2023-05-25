import torch.nn as nn 
import torch
import torch.nn.functional as F 

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

from pdb import set_trace as st 

class Args:
    image_size=256
    num_channels=256
    num_res_blocks=2
    num_heads=4
    num_heads_upsample=-1
    num_head_channels=64
    attention_resolutions="32,16,8"
    channel_mult=""
    dropout=0.0
    class_cond=False
    use_checkpoint=False
    use_scale_shift_norm=True
    resblock_updown=True
    use_fp16=False
    use_new_attention_order=False
    clip_denoised=True
    num_samples=10000
    batch_size=16
    use_ddim=False
    model_path=""
    classifier_path=""
    classifier_scale=1.0
    learn_sigma=True
    diffusion_steps=1000
    noise_schedule="linear"
    timestep_respacing=None
    use_kl=False
    predict_xstart=False
    rescale_timesteps=False
    rescale_learned_sigmas=False


class DiffusionRobustModel(nn.Module):
    def __init__(self, classifier, noise_sd, num_noise_vec=40, no_diffusion=False, dataset="trojai"):
        super().__init__()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            # torch.load("/home/mingjies/projects/diffusion_robustness/imagenet/256x256_diffusion_uncond.pt")
            torch.load("weights/256x256_diffusion_uncond.pt")
        )
        model.eval().cuda()

        model = torch.nn.DataParallel(model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()

        self.model = model 
        self.diffusion = diffusion 

        self.classifier = classifier
        self.dataset = dataset 
        self.no_diffusion = no_diffusion 
        self.num_noise_vec = num_noise_vec

        ## compute the timestep t corresponding to the added noise level according to https://arxiv.org/abs/2206.10550 
        real_sigma = 0
        t = 0
        while real_sigma < noise_sd * 2:
            t += 1
            a = diffusion.sqrt_alphas_cumprod[t]
            b = diffusion.sqrt_one_minus_alphas_cumprod[t]
            real_sigma = b / a

        self.sigma = noise_sd
        self.t = t 
        print("t found for sigma %.2f: %d"%(noise_sd, t))

    def reset_sigma(self, noise_sd):
        real_sigma = 0
        t = 0
        while real_sigma < noise_sd * 2:
            t += 1
            a = self.diffusion.sqrt_alphas_cumprod[t]
            b = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
            real_sigma = b / a

        self.sigma = noise_sd
        self.t = t 
        print("reset t for sigma %.2f: %d"%(noise_sd, t))

    def forward(self, x):
        x = x.repeat((self.num_noise_vec,1,1,1))

        if self.no_diffusion: # w/o diffusion
            x += torch.randn_like(x) * self.sigma 
        else: # w diffusion 
            x = x * 2 - 1
            x = self.diffusion_denoise(x, self.t)
            x = (x+1)/2

        out = self.classifier(x)

        if self.num_noise_vec == 1 and x.shape[0] != 1:
            return out 
        out = F.softmax(out, dim=1)

        out = torch.mean(out, dim=0, keepdims=True)
        return out

    def diffusion_denoise(self, x_start, t, multistep=False):
        t_batch = torch.tensor([t] * len(x_start)).cuda()

        noise = torch.randn_like(x_start)

        # Gaussian noise is added at this step
        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        with torch.enable_grad():
        # with torch.no_grad():
            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    print(i)
                    t_batch = torch.tensor([i] * len(x_start)).cuda()
                    out = self.diffusion.p_sample(
                        self.model,
                        out,
                        t_batch,
                        clip_denoised=True
                    )['sample']
            else:
                out = self.diffusion.p_sample(
                    self.model,
                    x_t_start,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        return out