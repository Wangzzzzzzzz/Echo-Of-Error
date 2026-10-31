import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Dict, Tuple


# ======================================================================
# Core JVP helper — architecture-agnostic
# ======================================================================

def _jvp_wrt_input(
    model: nn.Module,
    x: torch.Tensor,
    tangent: torch.Tensor,
    extra_args: tuple = (),
    jvp_bn_policy: str = "subset_train_stats",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (f(x), J_x @ tangent), retaining gradients to model parameters.

    Forward-mode AD differentiates the input, while functional_call keeps
    the live model parameters connected for the subsequent loss.backward().

    Args:
        model:      any nn.Module.
        x:          input tensor (will be detached internally for the JVP;
                    the live params keep the graph alive).
        tangent:    the tangent vector (typically the detached residual r).
        extra_args: additional positional arguments to model.forward, passed
                    through unchanged and NOT differentiated.
        jvp_bn_policy:
                    ``subset_train_stats`` is the default:
                    training-mode functional BN uses the current checker
                    subset's statistics but does not update its EMA buffers.
                    ``running_eval_stats`` temporarily makes only those BN
                    leaves use their stored running statistics. The rest of
                    the model (including dropout) remains in training mode.
    Returns:
        Model output and input JVP under the selected BN policy. Both have
        the model's output shape; the JVP retains parameter gradients.
    """
    valid_bn_policies = {"subset_train_stats", "running_eval_stats"}
    if jvp_bn_policy not in valid_bn_policies:
        raise ValueError(
            "jvp_bn_policy must be one of "
            f"{sorted(valid_bn_policies)}, got {jvp_bn_policy!r}"
        )

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    # Detach x and tangent so forward-mode AD is purely w.r.t. input,
    # not w.r.t. anything upstream.  The *params* remain attached to the
    # computation graph, so the JVP output is differentiable w.r.t. W.
    x_detached = x.detach()
    tangent_detached = tangent.detach()

    # Freeze running-stat updates in JVP-compatible BN leaves so their
    # in-place EMA mutations do not collide with the functorch transform.
    # Save both flags exactly: this keeps nested calls safe and guarantees
    # restoration when the transformed forward raises.
    _bn_states = [
        (module, module.training, module._freeze_stats)
        for module in model.modules()
        if (
            hasattr(module, "_freeze_stats")
            and hasattr(module, "running_mean")
            and hasattr(module, "running_var")
        )
    ]
    for module, training, _ in _bn_states:
        module._freeze_stats = True
        if jvp_bn_policy == "running_eval_stats":
            # Set only the BN leaf's flag. Calling model.eval() (or a parent
            # wrapper's eval()) would incorrectly disable dropout elsewhere.
            module.training = False
        else:
            module.training = training

    def func(inp):
        return torch.func.functional_call(
            model, {**params, **buffers}, (inp, *extra_args)
        )

    try:
        primal_out, tangent_out = torch.func.jvp(
            func, (x_detached,), (tangent_detached,)
        )
    finally:
        for module, training, freeze_stats in reversed(_bn_states):
            module.training = training
            module._freeze_stats = freeze_stats

    return primal_out, tangent_out


# ======================================================================
# Shared utilities
# ======================================================================

def _sample_residual(x):
    """Sample one random residual window per (batch, channel)."""
    B, C, T, nS = x.shape
    indices = torch.randint(0, nS, (B, C), device=x.device)
    indices = indices.view(B, C, 1, 1).expand(-1, -1, T, -1)
    return torch.gather(x, dim=3, index=indices).squeeze(-1)


def _adjust_tangent(tangent, Tx, Tr):
    """Reshape tangent from length Tr to Tx."""
    if Tx == Tr:
        return tangent
    elif Tx > Tr:
        tangent = (
            F.pad(tangent.permute(0, 2, 1), pad=(Tx - Tr, Tx - Tr))
             .unfold(-1, Tx, 1)
             .permute(0, 1, 3, 2)  # [B, C, Tx, nS]
        )
        return _sample_residual(tangent).permute(0, 2, 1)
    else:  # Tx < Tr
        return tangent[:, :Tx, :]


# ======================================================================
# Wrapper — general residual check for any nn.Module
# ======================================================================

class _CheckerBatchMixin:
    """Sample only checker rows; keep the forecasting loss full-batch.

    A positive absolute size takes precedence over the sampling ratio.
    Full-batch mode consumes no sampling RNG.
    """

    def _init_checker_batch(self, configs):
        self.checker_batch_ratio = float(getattr(configs, "checker_batch_ratio", 1.0))
        raw_size = getattr(configs, "checker_batch_size", 0)
        self.checker_batch_size = 0 if raw_size is None else int(raw_size)
        if not 0 < self.checker_batch_ratio <= 1:
            raise ValueError("checker_batch_ratio must be in (0, 1]")
        if self.checker_batch_size < 0 or (
            raw_size is not None and self.checker_batch_size != raw_size
        ):
            raise ValueError("checker_batch_size must be a nonnegative integer")
        self.checker_batch_seed = int(getattr(configs, "seed", 0))
        self._checker_batch_generators = {}
        self.jvp_bn_policy = getattr(configs, "jvp_bn_policy", "subset_train_stats")
        if self.jvp_bn_policy not in {"subset_train_stats", "running_eval_stats"}:
            raise ValueError("invalid jvp_bn_policy")

    def _checker_batch_indices(self, batch_size, device):
        """Return selected row indices, or None for the unchanged full batch.

        The private per-device generator leaves global training RNG untouched.
        Full-batch mode does not even create or advance a sampling generator.
        """
        if self.checker_batch_size > 0:
            size = self.checker_batch_size
        else:
            size = max(1, math.ceil(batch_size * self.checker_batch_ratio))
        if size >= batch_size:
            return None
        key = (device.type, device.index)
        generator = self._checker_batch_generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.checker_batch_seed)
            self._checker_batch_generators[key] = generator
        return torch.randperm(batch_size, device=device, generator=generator)[:size]

    @staticmethod
    def _select_checker_batch(indices, *tensors):
        """Select matching rows from input/residual tensors shaped [B, T, C]."""
        if indices is None:
            return tensors
        return tuple(tensor.index_select(0, indices) for tensor in tensors)


class ResidualCheckGeneral(_CheckerBatchMixin, nn.Module):
    """JVP-based residual check wrapper for any nn.Module."""

    def __init__(self, configs, Model):
        super(ResidualCheckGeneral, self).__init__()
        self._init_checker_batch(configs)

        self.IN = configs.instance_norm
        self.lambda_ = configs.regu_coef
        self.detaching = configs.detaching
        print(f"[ResCheckGeneral] lambda_res={self.lambda_}, "
              f"detaching={self.detaching}")

        self.padding_compensator = max(configs.seq_len / configs.pred_len, 1.0)

        configs_base = deepcopy(configs)
        configs_base.instance_norm = False
        self.base_model = Model(configs_base)

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.base_model.load_state_dict(state_dict, strict, assign)

    def state_dict(self):
        return self.base_model.state_dict()

    def parameters(self, recurse=True):
        return self.base_model.parameters(recurse)

    # ------------------------------------------------------------------
    # JVP computation  J_x · r  (architecture-agnostic)
    # ------------------------------------------------------------------
    def _compute_jvp(self, x: torch.Tensor, tangent: torch.Tensor):
        _, jvp_out = _jvp_wrt_input(
            self.base_model, x, tangent, jvp_bn_policy=self.jvp_bn_policy
        )
        return jvp_out

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def forward(self, *args):
        return self.forward_train(*args) if self.training else self.forward_test(*args)

    def forward_test(self, x, *args):
        if self.IN:
            x_mean = x.mean(dim=1, keepdim=True)
            x = x - x_mean
        y = self.base_model(x)
        if self.IN:
            y = y + x_mean
        return y

    def forward_train(self, x, y, *args):
        if self.IN:
            x_mean = x.mean(dim=1, keepdim=True)
            x = x - x_mean
            y = y - x_mean

        B, Tx, C = x.shape
        _, Ty, _ = y.shape

        # --- 1. normal forward ---
        yf_hat = self.base_model(x)

        # --- 2. residual ---
        if self.detaching:
            residual_ = (y - yf_hat).detach()
        else:
            residual_ = y - yf_hat

        # --- 3. adjust residual length to match Tx ---
        Tr = residual_.shape[1]
        residual_ = _adjust_tangent(residual_, Tx, Tr)

        # --- 4. JVP-based checker: J_x(W) · residual ---
        indices = self._checker_batch_indices(B, x.device)
        x_checker, residual_checker = self._select_checker_batch(indices, x, residual_)
        val_y = self._compute_jvp(x_checker, residual_checker)

        # --- losses ---
        floss = F.mse_loss(yf_hat, y)
        val_floss = F.mse_loss(val_y, torch.zeros_like(val_y))
        loss_ = floss + val_floss * self.lambda_ * self.padding_compensator

        with torch.no_grad():
            pred_norm = (yf_hat ** 2).mean()

        return {"loss": loss_, "mse": floss, "checker": val_floss,
                "pred_norm": pred_norm}


# ======================================================================
# Wrapper — TF-FR consistency check for any nn.Module (single model)
# ======================================================================

class TFCheckGeneral(nn.Module):
    """TF-FR consistency check wrapper. Penalises ||f(adj(y)) - f(adj(y_hat))||^2."""

    def __init__(self, configs, Model):
        super(TFCheckGeneral, self).__init__()

        self.IN = configs.instance_norm
        self.lambda_ = configs.regu_coef
        self.detaching = configs.detaching
        print(f"[TFCheckGeneral] lambda_tf={self.lambda_}, "
              f"detaching={self.detaching}")

        self.padding_compensator = max(configs.seq_len / configs.pred_len, 1.0)

        configs_base = deepcopy(configs)
        configs_base.instance_norm = False
        self.base_model = Model(configs_base)

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.base_model.load_state_dict(state_dict, strict, assign)

    def state_dict(self):
        return self.base_model.state_dict()

    def parameters(self, recurse=True):
        return self.base_model.parameters(recurse)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def forward(self, *args):
        return self.forward_train(*args) if self.training else self.forward_test(*args)

    def forward_test(self, x, *args):
        if self.IN:
            x_mean = x.mean(dim=1, keepdim=True)
            x = x - x_mean
        y = self.base_model(x)
        if self.IN:
            y = y + x_mean
        return y

    def forward_train(self, x, y, *args):
        if self.IN:
            x_mean = x.mean(dim=1, keepdim=True)
            x = x - x_mean
            y = y - x_mean

        B, Tx, C = x.shape
        _, Ty, _ = y.shape

        # --- 1. normal forward ---
        yf_hat = self.base_model(x)

        # --- 2. TF and FR inputs ---
        if self.detaching:
            y_fr_input = yf_hat.detach()
        else:
            y_fr_input = yf_hat

        y_tf = _adjust_tangent(y, Tx, Ty)
        y_fr = _adjust_tangent(y_fr_input, Tx, Ty)

        # --- 3. TF-FR consistency: ‖f(adj(y)) − f(adj(ŷ))‖² ---
        y_from_tf = self.base_model(y_tf)
        y_from_fr = self.base_model(y_fr)
        tf_fr_diff = y_from_tf - y_from_fr

        # --- losses ---
        floss = F.mse_loss(yf_hat, y)
        val_floss = F.mse_loss(tf_fr_diff, torch.zeros_like(tf_fr_diff))
        loss_ = floss + val_floss * self.lambda_ * self.padding_compensator

        with torch.no_grad():
            pred_norm = (yf_hat ** 2).mean()

        return {"loss": loss_, "mse": floss, "checker": val_floss,
                "pred_norm": pred_norm}


# ======================================================================
# Shared branch-model helpers (linear shortcut delegation)
# ======================================================================

class _BranchBase(nn.Module):
    """Shared plumbing for MLP + Linear shortcut wrappers."""

    def _init_shortcut(self, configs):
        if self.individual:
            self.linear_shortcut = nn.ModuleList([
                nn.Linear(configs.seq_len, configs.pred_len, bias=False)
                for _ in range(configs.enc_in)
            ])
        else:
            self.linear_shortcut = nn.Linear(
                configs.seq_len, configs.pred_len, bias=False
            )

    def parameters(self, recurse=True):
        yield from self.mlp_model.parameters(recurse)
        yield from self.linear_shortcut.parameters(recurse)

    def state_dict(self):
        state = {}
        for k, v in self.mlp_model.state_dict().items():
            state[k] = v
        for k, v in self.linear_shortcut.state_dict().items():
            state[f"_shortcut.{k}"] = v
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mlp_state, lin_state = {}, {}
        for k, v in state_dict.items():
            if k.startswith("_shortcut."):
                lin_state[k[len("_shortcut."):]] = v
            else:
                mlp_state[k] = v
        self.mlp_model.load_state_dict(mlp_state, strict=strict, assign=assign)
        if lin_state:
            self.linear_shortcut.load_state_dict(
                lin_state, strict=strict, assign=assign
            )

    def _linear_forward(self, x):
        """x: [B, T_in, C] → [B, T_out, C]."""
        h = x.permute(0, 2, 1)  # [B, C, T]
        if self.individual:
            y = torch.cat(
                [self.linear_shortcut[i](h[:, [i], :])
                 for i in range(h.shape[1])],
                dim=1,
            )
        else:
            y = self.linear_shortcut(h)
        return y.permute(0, 2, 1)

    def forward(self, *args):
        return self.forward_train(*args) if self.training else self.forward_test(*args)

    def forward_test(self, x, *args):
        if self.IN:
            x_mean = x.mean(dim=1, keepdim=True)
            x = x - x_mean
        y = self.mlp_model(x) + self._linear_forward(x)
        if self.IN:
            y = y + x_mean
        return y


# ======================================================================
# BranchResidualCheck — MLP + Linear shortcut with JVP residual check
# ======================================================================

class BranchResidualCheck(_CheckerBatchMixin, _BranchBase):
    """MLP + Linear shortcut with JVP-based residual check.

    decouple_checker modes: none (combined), decouple (separate norms),
    isolate (branch-specific residuals).
    """

    def __init__(self, configs, MLPModel):
        super(BranchResidualCheck, self).__init__()
        self._init_checker_batch(configs)

        self.IN = configs.instance_norm
        self.lambda_res = configs.regu_coef
        self.detaching = configs.detaching
        self.individual = getattr(configs, 'individual', False)
        self.decouple_checker = getattr(configs, 'decouple_checker', 'none')

        _mlp_coef = getattr(configs, 'regu_coef_mlp', None)
        self.lambda_res_mlp = _mlp_coef if _mlp_coef is not None else self.lambda_res

        print(f"[BranchResidualCheck] lambda_res={self.lambda_res}, "
              f"lambda_res_mlp={self.lambda_res_mlp}, "
              f"detaching={self.detaching}, individual={self.individual}, "
              f"decouple_checker={self.decouple_checker}")

        self.padding_compensator = max(configs.seq_len / configs.pred_len, 1.0)

        configs_base = deepcopy(configs)
        configs_base.instance_norm = False

        self.mlp_model = MLPModel(configs_base)
        self._init_shortcut(configs)

    def forward_train(self, x, y, *args):
        if self.IN:
            x_mean = x.mean(dim=1, keepdim=True)
            x = x - x_mean
            y = y - x_mean

        B, Tx, C = x.shape
        _, Ty, _ = y.shape

        # --- 1. forward both branches ---
        y_mlp = self.mlp_model(x)
        y_lin = self._linear_forward(x)
        y_hat = y_mlp + y_lin

        floss = F.mse_loss(y_hat, y)
        loss_ = floss

        with torch.no_grad():
            mlp_norm = (y_mlp ** 2).mean()
            lin_norm = (y_lin ** 2).mean()
            mlp_frac = mlp_norm / (mlp_norm + lin_norm + 1e-8)

        out = {"loss": floss, "mse": floss,
               "mlp_norm": mlp_norm, "lin_norm": lin_norm,
               "mlp_frac": mlp_frac}

        # ==============================================================
        # Residual check
        # ==============================================================
        if self.lambda_res > 0:
            checker_indices = self._checker_batch_indices(B, x.device)
            if self.decouple_checker == 'isolate':
                # Branch-specific residuals
                if self.detaching:
                    residual_lin = (y - y_lin).detach()
                    residual_all = (y - y_hat).detach()
                else:
                    residual_lin = y - y_lin
                    residual_all = y - y_hat

                # Linear branch
                res_lin = _adjust_tangent(residual_lin, Tx, residual_lin.shape[1])
                res_all = _adjust_tangent(residual_all, Tx, residual_all.shape[1])
                x_checker, res_lin, res_all = self._select_checker_batch(
                    checker_indices, x, res_lin, res_all
                )
                res_jvp_lin = self._linear_forward(res_lin)
                checker_loss_lin = F.mse_loss(res_jvp_lin, torch.zeros_like(res_jvp_lin))

                # Both branch checkers use the same subset; MSE above does not.
                _, res_jvp_mlp = _jvp_wrt_input(
                    self.mlp_model, x_checker, res_all, jvp_bn_policy=self.jvp_bn_policy
                )
                checker_loss_mlp = F.mse_loss(res_jvp_mlp, torch.zeros_like(res_jvp_mlp))

                loss_ = (loss_
                         + checker_loss_lin * self.lambda_res * self.padding_compensator
                         + checker_loss_mlp * self.lambda_res_mlp * self.padding_compensator)
                out["checker_lin"] = checker_loss_lin
                out["checker_mlp"] = checker_loss_mlp
                out["checker"] = checker_loss_lin + checker_loss_mlp

            else:
                # Shared residual for 'none' and 'decouple'
                if self.detaching:
                    residual_ = (y - y_hat).detach()
                else:
                    residual_ = y - y_hat

                residual_ = _adjust_tangent(residual_, Tx, residual_.shape[1])
                x_checker, residual_ = self._select_checker_batch(checker_indices, x, residual_)
                _, res_jvp_mlp = _jvp_wrt_input(
                    self.mlp_model, x_checker, residual_, jvp_bn_policy=self.jvp_bn_policy
                )
                res_jvp_lin = self._linear_forward(residual_)

                if self.decouple_checker == 'decouple':
                    checker_loss_lin = F.mse_loss(res_jvp_lin, torch.zeros_like(res_jvp_lin))
                    checker_loss_mlp = F.mse_loss(res_jvp_mlp, torch.zeros_like(res_jvp_mlp))
                    loss_ = (loss_
                             + checker_loss_lin * self.lambda_res * self.padding_compensator
                             + checker_loss_mlp * self.lambda_res_mlp * self.padding_compensator)
                    out["checker_lin"] = checker_loss_lin
                    out["checker_mlp"] = checker_loss_mlp
                    out["checker"] = checker_loss_lin + checker_loss_mlp
                else:
                    res_jvp = res_jvp_mlp + res_jvp_lin
                    checker_loss = F.mse_loss(res_jvp, torch.zeros_like(res_jvp))
                    loss_ = loss_ + checker_loss * self.lambda_res * self.padding_compensator
                    out["checker"] = checker_loss

        out["loss"] = loss_
        return out


# ======================================================================
# BranchTFCheck — MLP + Linear shortcut with TF-FR consistency check
# ======================================================================

class BranchTFCheck(_BranchBase):
    """MLP + Linear shortcut with TF-FR consistency check.

    decouple_checker modes: none (combined), decouple (separate norms),
    isolate (branch-specific residuals).
    """

    def __init__(self, configs, MLPModel):
        super(BranchTFCheck, self).__init__()

        self.IN = configs.instance_norm
        self.lambda_res = configs.regu_coef
        self.detaching = configs.detaching
        self.individual = getattr(configs, 'individual', False)
        self.decouple_checker = getattr(configs, 'decouple_checker', 'none')

        _mlp_coef = getattr(configs, 'regu_coef_mlp', None)
        self.lambda_res_mlp = _mlp_coef if _mlp_coef is not None else self.lambda_res

        print(f"[BranchTFCheck] lambda_res={self.lambda_res}, "
              f"lambda_res_mlp={self.lambda_res_mlp}, "
              f"detaching={self.detaching}, individual={self.individual}, "
              f"decouple_checker={self.decouple_checker}")

        self.padding_compensator = max(configs.seq_len / configs.pred_len, 1.0)

        configs_base = deepcopy(configs)
        configs_base.instance_norm = False

        self.mlp_model = MLPModel(configs_base)
        self._init_shortcut(configs)

    def forward_train(self, x, y, *args):
        if self.IN:
            x_mean = x.mean(dim=1, keepdim=True)
            x = x - x_mean
            y = y - x_mean

        B, Tx, C = x.shape
        _, Ty, _ = y.shape

        # --- 1. forward both branches ---
        y_mlp = self.mlp_model(x)
        y_lin = self._linear_forward(x)
        y_hat = y_mlp + y_lin

        floss = F.mse_loss(y_hat, y)
        loss_ = floss

        with torch.no_grad():
            mlp_norm = (y_mlp ** 2).mean()
            lin_norm = (y_lin ** 2).mean()
            mlp_frac = mlp_norm / (mlp_norm + lin_norm + 1e-8)

        out = {"loss": floss, "mse": floss,
               "mlp_norm": mlp_norm, "lin_norm": lin_norm,
               "mlp_frac": mlp_frac}

        # ==============================================================
        # TF-FR consistency check
        # ==============================================================
        if self.lambda_res > 0:
            if self.decouple_checker == 'isolate':
                # Branch-specific residuals
                if self.detaching:
                    residual_lin = (y - y_lin).detach()
                    y_fr_input = y_hat.detach()
                else:
                    residual_lin = y - y_lin
                    y_fr_input = y_hat

                # Linear branch: f_lin(r_lin_adj)
                res_lin = _adjust_tangent(residual_lin, Tx, residual_lin.shape[1])
                tf_fr_lin = self._linear_forward(res_lin)
                checker_loss_lin = F.mse_loss(tf_fr_lin, torch.zeros_like(tf_fr_lin))

                # MLP branch: f_mlp(adj(y)) - f_mlp(adj(ŷ))
                y_tf = _adjust_tangent(y, Tx, Ty)
                y_fr = _adjust_tangent(y_fr_input, Tx, Ty)
                tf_fr_mlp = self.mlp_model(y_tf) - self.mlp_model(y_fr)
                checker_loss_mlp = F.mse_loss(tf_fr_mlp, torch.zeros_like(tf_fr_mlp))

                loss_ = (loss_
                         + checker_loss_lin * self.lambda_res * self.padding_compensator
                         + checker_loss_mlp * self.lambda_res_mlp * self.padding_compensator)
                out["checker_lin"] = checker_loss_lin
                out["checker_mlp"] = checker_loss_mlp
                out["checker"] = checker_loss_lin + checker_loss_mlp

            else:
                # Shared residual for 'none' and 'decouple'
                if self.detaching:
                    residual_ = (y - y_hat).detach()
                    y_fr_input = y_hat.detach()
                else:
                    residual_ = y - y_hat
                    y_fr_input = y_hat

                residual_ = _adjust_tangent(residual_, Tx, residual_.shape[1])

                # MLP: TF-FR difference via two forward passes
                y_tf = _adjust_tangent(y, Tx, Ty)
                y_fr = _adjust_tangent(y_fr_input, Tx, Ty)
                tf_fr_mlp = self.mlp_model(y_tf) - self.mlp_model(y_fr)

                # Linear: f_lin(r_adj)
                tf_fr_lin = self._linear_forward(residual_)

                if self.decouple_checker == 'decouple':
                    checker_loss_lin = F.mse_loss(tf_fr_lin, torch.zeros_like(tf_fr_lin))
                    checker_loss_mlp = F.mse_loss(tf_fr_mlp, torch.zeros_like(tf_fr_mlp))
                    loss_ = (loss_
                             + checker_loss_lin * self.lambda_res * self.padding_compensator
                             + checker_loss_mlp * self.lambda_res_mlp * self.padding_compensator)
                    out["checker_lin"] = checker_loss_lin
                    out["checker_mlp"] = checker_loss_mlp
                    out["checker"] = checker_loss_lin + checker_loss_mlp
                else:
                    tf_fr = tf_fr_mlp + tf_fr_lin
                    checker_loss = F.mse_loss(tf_fr, torch.zeros_like(tf_fr))
                    loss_ = loss_ + checker_loss * self.lambda_res * self.padding_compensator
                    out["checker"] = checker_loss

        out["loss"] = loss_
        return out


# ======================================================================
# Monitoring-only wrapper — no regularisation, logs checker diagnostics
# ======================================================================

class DummyCheckGeneral(nn.Module):
    """Plain MSE training with JVP residual check logged for diagnostics."""

    def __init__(self, configs, Model):
        super(DummyCheckGeneral, self).__init__()

        self.IN = configs.instance_norm
        print(f"[DummyCheckGeneral] mode=base")

        configs_base = deepcopy(configs)
        configs_base.instance_norm = False
        self.base_model = Model(configs_base)

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.base_model.load_state_dict(state_dict, strict=strict, assign=assign)

    def state_dict(self):
        return self.base_model.state_dict()

    def parameters(self, recurse=True):
        yield from self.base_model.parameters(recurse)

    # ------------------------------------------------------------------
    def forward(self, *args):
        return self.forward_train(*args) if self.training else self.forward_test(*args)

    def forward_test(self, x, *args):
        if self.IN:
            x_mean = x.mean(dim=1, keepdim=True)
            x = x - x_mean
        y = self.base_model(x)
        if self.IN:
            y = y + x_mean
        return y

    def forward_train(self, x, y, *args):
        if self.IN:
            x_mean = x.mean(dim=1, keepdim=True)
            x = x - x_mean
            y = y - x_mean

        B, Tx, C = x.shape
        _, Ty, _ = y.shape

        y_hat = self.base_model(x)
        floss = F.mse_loss(y_hat, y)
        out = {"loss": floss}

        with torch.no_grad():
            out["base_norm"] = (y_hat ** 2).mean()

            # Residual check monitoring
            residual_ = (y - y_hat).detach()
            residual_ = _adjust_tangent(residual_, Tx, residual_.shape[1])
            _, res_jvp = _jvp_wrt_input(self.base_model, x, residual_)
            out["checker_jvp"] = F.mse_loss(res_jvp, torch.zeros_like(res_jvp))

        return out
