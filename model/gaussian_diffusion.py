import enum
import math
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn


class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, mean_type, noise_schedule, noise_scale, noise_min, noise_max,
                 steps, device, history_num_per_term=10, beta_fixed=True):

        super(GaussianDiffusion, self).__init__()
        self.denoise_fn = denoise_fn
        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device

        self.history_num_per_term = history_num_per_term
        self.Lt_history = th.zeros(steps, history_num_per_term, dtype=th.float64).to(device)
        self.Lt_count = th.zeros(steps, dtype=int).to(device)

        if noise_scale != 0.:
            self.betas = th.tensor(self.get_betas(), dtype=th.float64).to(self.device)
            if beta_fixed:
                if len(self.betas) > 0:
                    self.betas[0] = 0.00001
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"
            self.calculate_for_diffusion()
        else:
            if self.steps > 0:
                self.betas = th.zeros(steps, dtype=th.float64).to(self.device)
                self.calculate_for_diffusion()
            else:
                self.alphas_cumprod = th.tensor([]).to(self.device)
                self.alphas_cumprod_prev = th.tensor([]).to(self.device)
                self.sqrt_alphas_cumprod = th.tensor([]).to(self.device)
                self.sqrt_one_minus_alphas_cumprod = th.tensor([]).to(self.device)
                self.posterior_variance = th.tensor([]).to(self.device)
                self.posterior_log_variance_clipped = th.tensor([]).to(self.device)
                self.posterior_mean_coef1 = th.tensor([]).to(self.device)
                self.posterior_mean_coef2 = th.tensor([]).to(self.device)

    def get_betas(self):
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
                self.steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            )
        elif self.noise_schedule == "binomial":
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")

    def calculate_for_diffusion(self):
        if len(self.betas) == 0:
            self.alphas_cumprod = th.tensor([1.0]).to(self.device)
            self.alphas_cumprod_prev = th.tensor([1.0]).to(self.device)
            self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)
            self.log_one_minus_alphas_cumprod = th.log(1.0 - self.alphas_cumprod)
            if (1.0 - self.alphas_cumprod == 0).any(): self.log_one_minus_alphas_cumprod = th.full_like(
                self.alphas_cumprod, -float('inf'))
            self.sqrt_recip_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod)
            self.sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod - 1)
            self.posterior_variance = th.zeros_like(self.betas)
            self.posterior_log_variance_clipped = th.log(th.zeros_like(self.betas) + 1e-20)
            self.posterior_mean_coef1 = th.zeros_like(self.betas)
            self.posterior_mean_coef2 = th.zeros_like(self.betas)
            return

        alphas = 1.0 - self.betas
        self.alphas_cumprod = th.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = th.cat([th.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)
        self.alphas_cumprod_next = th.cat([self.alphas_cumprod[1:], th.tensor([0.0]).to(self.device)]).to(self.device)
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)

        log_val = 1.0 - self.alphas_cumprod
        self.log_one_minus_alphas_cumprod = th.log(th.where(log_val > 0, log_val, th.full_like(log_val, 1e-20)))

        self.sqrt_recip_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        if len(self.posterior_variance) > 0: self.posterior_variance[0] = self.betas[0]

        self.posterior_log_variance_clipped = th.log(
            th.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]) if len(
                self.posterior_variance) > 1 else th.tensor(
                [th.log(self.betas[0] if len(self.betas) > 0 else 1e-20)]).to(self.device)
        )

        self.posterior_mean_coef1 = (
                self.betas * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * th.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def p_sample(self, x_start, x_mashup_2hop, x_contextual_api_api_sim, x_api_compl, x_mashup_text_sim, steps,
                 sampling_noise=False):
        if steps == 0:
            return x_start, None, []

        t_init = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
        x_t = self.q_sample(x_start, t_init)

        indices = list(range(steps))[::-1]

        all_dynamic_weights_over_steps = []
        last_active_branch_names = []

        if self.noise_scale == 0.:
            for i in indices:
                t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
                model_pred, step_weights, step_names = self.denoise_fn(x_t, x_mashup_2hop, x_contextual_api_api_sim,
                                                                       x_api_compl, x_mashup_text_sim, t)
                x_t = model_pred
                if step_weights is not None: all_dynamic_weights_over_steps.append(step_weights.detach().cpu())
                if i == 0: last_active_branch_names = step_names
            return x_t, all_dynamic_weights_over_steps, last_active_branch_names

        for i in indices:
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
            out_dict, step_weights, step_names = self.p_mean_variance(x_t, x_mashup_2hop, x_contextual_api_api_sim,
                                                                      x_api_compl, x_mashup_text_sim, t)

            if step_weights is not None: all_dynamic_weights_over_steps.append(step_weights.detach().cpu())
            if i == 0: last_active_branch_names = step_names

            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )
                x_t = out_dict["mean"] + nonzero_mask * th.exp(0.5 * out_dict["log_variance"]) * noise
            else:
                x_t = out_dict["mean"]
        return x_t, all_dynamic_weights_over_steps, last_active_branch_names

    def training_losses(self, x_start, x_mashup_2hop, x_contextual_api_api_sim, x_api_compl, x_mashup_text_sim,
                        reweight=False):
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, 'importance' if hasattr(self, 'Lt_count') and (
                    self.Lt_count == self.history_num_per_term).all() else 'uniform')

        noise = th.randn_like(x_start)
        if self.noise_scale != 0. and self.steps > 0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        terms = {}
        denoised_output_for_loss, dynamic_weights, active_branch_names = self.denoise_fn(x_t, x_mashup_2hop,
                                                                                         x_contextual_api_api_sim,
                                                                                         x_api_compl, x_mashup_text_sim,
                                                                                         ts)

        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert denoised_output_for_loss.shape == target.shape == x_start.shape, \
            f"Shape mismatch: output={denoised_output_for_loss.shape}, target={target.shape}, x_start={x_start.shape}"

        mse = mean_flat((target - denoised_output_for_loss) ** 2)

        loss = mse
        if reweight and self.steps > 0:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                safe_ts = th.clamp(ts, 0, self.steps - 1)
                likelihood = mean_flat(
                    (x_start - self._predict_xstart_from_eps(x_t, safe_ts, denoised_output_for_loss)) ** 2 / 2.0)
                loss = th.where((safe_ts == 0), likelihood, mse)
                weight = th.ones_like(loss)
        else:
            weight = th.ones_like(loss)

        terms["loss"] = weight * loss

        if hasattr(self, 'Lt_history') and self.steps > 0:
            for i_sample in range(len(ts)):
                t_scalar = ts[i_sample].item()
                loss_scalar = terms["loss"][i_sample].detach()

                if self.Lt_count[t_scalar] == self.history_num_per_term:
                    self.Lt_history[t_scalar, :-1] = self.Lt_history[t_scalar, 1:].clone()
                    self.Lt_history[t_scalar, -1] = loss_scalar
                else:
                    self.Lt_history[t_scalar, self.Lt_count[t_scalar]] = loss_scalar
                    self.Lt_count[t_scalar] += 1

        if not th.all(pt == 1.0):
            terms["loss"] = terms["loss"] / pt

        if terms["loss"].ndim > 0:
            terms["loss"] = terms["loss"].mean()

        return terms, dynamic_weights, active_branch_names

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance' and self.steps > 0:
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')

            Lt_sqrt = th.sqrt(th.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / th.sum(Lt_sqrt)
            pt_all = th.nan_to_num(pt_all, nan=1.0 / self.steps)

            pt_all = pt_all * (1.0 - uniform_prob) + uniform_prob / self.steps

            assert abs(pt_all.sum(-1).item() - 1.) < 1e-4, f"Probabilities do not sum to 1: {pt_all.sum(-1).item()}"

            t = th.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * self.steps
            return t, pt

        elif method == 'uniform' or self.steps == 0:
            if self.steps == 0:
                t = th.zeros(batch_size, device=device).long()
            else:
                t = th.randint(0, self.steps, (batch_size,), device=device).long()
            pt = th.ones_like(t).float()
            return t, pt
        else:
            raise ValueError("Invalid sampling method or steps=0 without uniform fallback")

    def q_sample(self, x_start, t, noise=None):
        if self.steps == 0: return x_start
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        if self.steps == 0:
            return x_start, th.zeros_like(x_start), th.zeros_like(x_start)

        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, x_mashup_2hop, x_contextual_api_api_sim, x_api_compl, x_mashup_text_sim, t):
        if self.steps == 0:
            return {"mean": x, "variance": th.zeros_like(x), "log_variance": th.full_like(x, -float('inf')),
                    "pred_xstart": x}, None, []

        B, *_ = x.shape
        assert t.shape == (B,)

        model_pred_for_loss, dynamic_weights, active_branch_names = self.denoise_fn(x, x_mashup_2hop,
                                                                                    x_contextual_api_api_sim,
                                                                                    x_api_compl, x_mashup_text_sim, t)

        model_variance = self._extract_into_tensor(self.posterior_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)

        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_pred_for_loss
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, model_pred_for_loss)
        else:
            raise NotImplementedError(self.mean_type)

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert model_mean.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }, dynamic_weights, active_branch_names

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        if self.steps == 0: return x_t
        return (
                self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def SNR(self, t):
        if self.steps == 0: return th.full_like(t.float(), float('inf'))
        safe_t = th.clamp(t, 0, self.steps - 1)
        return self.alphas_cumprod[safe_t] / (1 - self.alphas_cumprod[safe_t])

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        if self.steps == 0 and arr.numel() == 0:
            return th.ones(timesteps.shape[0], device=timesteps.device).view(-1, *([1] * (len(broadcast_shape) - 1)))

        res = arr.to(timesteps.device).gather(-1, timesteps).float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    # 对齐 DiffRec: 先得到 alpha_bar = 1 - variance，再逐步离散化得到 betas
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + th.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
