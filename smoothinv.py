import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional

from torch.optim.lr_scheduler import StepLR

from pdb import set_trace as st

# Modification of the code from https://github.com/Hadisalman/smoothing-adversarial
class SmoothInv():
    """
    SmoothInv
    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.
    """

    def __init__(self,
                 max_steps: int,
                 step_size: float,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cuda')) -> None:
        self.max_steps = max_steps
        self.step_size = step_size 
        self.max_norm = max_norm
        self.device = device

    def synthesize(self, model, inputs, labels) -> torch.Tensor:
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
        batch_size = labels.shape[0]
        delta = torch.zeros((len(labels), *inputs.shape[1:]), dtype=inputs.dtype, requires_grad=True, device=self.device)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=self.step_size, momentum=0.9)

        scheduler = StepLR(optimizer, step_size=300, gamma=0.5)

        for i in range(self.max_steps):
            adv = inputs + delta

            y_prob = model(adv)
            logsoftmax = torch.log(y_prob.clamp(min=1e-20))
            ce_loss = F.nll_loss(logsoftmax, labels)
            loss = ce_loss

            optimizer.zero_grad()
            loss.backward()

            if i % 100 == 0:
                print("iter %d prob %.8f prediction %d"%(i, loss.item(), y_prob.max(1)[1].item()))

            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad = torch.zeros_like(delta.grad)

            optimizer.step()
            scheduler.step()

            delta.data.add_(inputs)
            delta.data.clamp_(0, 1).sub_(inputs)

            delta.data.renorm_(p=2, dim=0, maxnorm=self.max_norm)

        return inputs + delta 