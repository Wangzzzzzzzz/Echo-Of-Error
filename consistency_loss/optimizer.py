import torch
import numpy as np

class LinProxyGradOptimizer(torch.optim.Optimizer):
    def __init__(self, params, base_optim, mode="global", **kwargs):
        assert mode in ["global", "local"]
        defaults = dict(**kwargs)
        super(LinProxyGradOptimizer, self).__init__(params, defaults)

        self.base_optimizer = base_optim(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.logged_loss = np.inf
        self.mode = mode

    @torch.no_grad()
    def step(self, cur_loss = None):
        if (self.mode == "local") and (cur_loss < self.logged_loss):
            flag_update_grad = True
            self.logged_loss = cur_loss
        else:
            flag_update_grad = False

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if ("lin_buf_" not in self.state[p]) or flag_update_grad:
                    self.state[p]["lin_buf_"] = p.grad

        
        # change to linearized gradient proxy
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.grad = self.state[p]["lin_buf_"]
        self.base_optimizer.step()



class WeightDecayOptimizer(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, l1_decay=0.0, l2_decay=0.0, **kwargs):
        if l1_decay < 0.0 or l2_decay < 0.0:
            raise ValueError("L1 and L2 decay coefficients must be non-negative")
        kwargs.pop("weight_decay", None)

        defaults = dict(l1_decay=l1_decay, l2_decay = l2_decay, **kwargs)
        super(WeightDecayOptimizer, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def step(self, closure=None):
        
        # Apply weight decay manually
        for group in self.base_optimizer.param_groups:
            lr_ = group["lr"]
            l1_decay_ = group['l1_decay']
            l2_decay_ = group['l2_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                if l1_decay_> 0:
                    p.add_(lr_*l1_decay_*torch.sign(p))
                
                # Apply L2 regularization (w^2)
                if l2_decay_ > 0:
                    p.add_(lr_*l2_decay_*p)

        # Perform the base optimizer's step
        self.base_optimizer.step(closure)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups