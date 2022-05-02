import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        restore_fn,
        denoise_fn,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        device = None
    ):
        super().__init__()
        self.restore_fn = restore_fn
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.device=device
        if schedule_opt is not None:
            self.set_new_noise_schedule(schedule_opt,device)
        if loss_type is not None:
            self.set_loss()

    def set_loss(self, device=None):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss()
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss()
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        if device is not None:
            to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        else:
            raise ValueError('Device not Found')

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, restore_step, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            x_restore = self.restore_fn(x_in, restore_step)[0]
            condition_x = torch.cat([x_in, x_restore],dim=1)
            ret_img = x_restore
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=condition_x)
                # if i % sample_inter == 0:
                    # img = img
            ret_img = torch.cat([ret_img, x_restore + img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1].unsqueeze(0)

    @torch.no_grad()
    def p_sample_skip(self, denoise_fn, xt, cond, t, next_t, eta=0):
        n = t.size(0)
        at = self.alphas_cumprod[(t+1).long()]
        at_next = self.alphas_cumprod[(next_t+1).long()]
        noise_level = torch.FloatTensor(
                [self.sqrt_alphas_cumprod_prev[t.long()]]).repeat(n, 1).to(t.device)
        et = denoise_fn(torch.cat([cond, xt],dim=1),noise_level)
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(xt) + c2 * et
        return xt_next

    @torch.no_grad()
    def p_sample_skiploop(self, x_in, restore_step, seq, eta = 0, continous=False):
        device = self.betas.device
        sample_steps = len(seq)
        sample_inter = (1 | (sample_steps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, sample_steps)), desc='sampling loop time step', total=sample_steps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            shape = x_in.shape
            xt = torch.randn(shape, device=device)
            x_restore = self.restore_fn(x_in, restore_step)[0]
            ret_img = x_restore
            # condition = torch.cat([x_in,x_restore],dim=1)
            condition = torch.cat([x_in, x_restore],dim=1)
            seq_next = [-1] + list(seq[:-1])
            n = x_in.size(0)
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sampling loop time step', total=sample_steps):
                t = (torch.ones(n) * i).to(device)
                next_t = (torch.ones(n) * j).to(device)
                xt_next = self.p_sample_skip(self.denoise_fn, xt, condition, t, next_t) 
                # at = self.alphas_cumprod[(t+1).long()]
                # at_next = self.alphas_cumprod[(next_t+1).long()]
                # noise_level = torch.FloatTensor(
                #         [self.sqrt_alphas_cumprod_prev[t.long()+1]]).repeat(n, 1).to(device)
                # et = self.denoise_fn(torch.cat([x_in, xt],dim=1),noise_level)
                 
                # x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                # c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                # c2 = ((1 - at_next) - c1 ** 2).sqrt()
                # xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x_in) + c2 * et
                xt = xt_next
                # if i % sample_inter == 0:
                    # img = img
            ret_img = torch.cat([ret_img, x_restore + xt_next], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1].unsqueeze(0)

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def restore(self, x_in, restore_step, continous=False):
        return self.p_sample_loop(x_in, restore_step, continous)

    @torch.no_grad()
    def skip_restore(self, x_in, restore_step, seq, continous=False):
        return self.p_sample_skiploop(x_in, restore_step, seq,continous=continous) 

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, input, restore, target, noise=None):
        x_start = target - restore
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([input, restore,  x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, input, restore, target):
        return self.p_losses(input, restore, target)
