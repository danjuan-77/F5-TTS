"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map
        
        # 注册 STFT 参数
        self.register_buffer('window', torch.hann_window(self.mel_spec.win_length))

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=True, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def istft(self, mag, phase_sin, phase_cos):
        """将预测的幅度谱和相位谱转换回音频"""
        # 重建复数形式的STFT
        phase = torch.atan2(phase_sin, phase_cos)
        mag = mag.exp()  # 从对数幅度谱转回线性幅度谱
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        spec_complex = torch.complex(real, imag)
        
        # 执行 iSTFT
        audio = torch.istft(
            spec_complex,
            n_fft=self.mel_spec.n_fft,
            hop_length=self.mel_spec.hop_length,
            win_length=self.mel_spec.win_length,
            window=self.window,
            center=True,
            return_complex=False
        )
        return audio

    def forward(self, batch, return_loss=False):
        if not return_loss:
            return self.sample(batch)

        mel = batch["mel_spec"]  # [B, 3, T, F]
        text = batch["text"]
        text_lengths = batch.get("text_lengths")
        mel_lengths = batch.get("mel_lengths")

        # 分离幅度谱和相位谱
        log_magnitude = mel[:, 0]  # [B, T, F]
        phase_sin = mel[:, 1]
        phase_cos = mel[:, 2]

        # 获取文本条件
        text_cond = self.get_text_cond(text, text_lengths)
        text_mask = lens_to_mask(text_lengths) if exists(text_lengths) else None

        # 生成预测
        pred = self.predict(log_magnitude, text_cond, text_mask, mel_lengths)  # [B, 3, T, F]
        
        # 分离预测的幅度谱和相位谱
        pred_magnitude = pred[:, 0]  # [B, T, F]
        pred_phase_sin = pred[:, 1]
        pred_phase_cos = pred[:, 2]

        # 计算频谱损失
        spec_loss = F.mse_loss(pred_magnitude, log_magnitude, reduction='none')
        phase_loss = F.mse_loss(pred_phase_sin, phase_sin, reduction='none') + \
                    F.mse_loss(pred_phase_cos, phase_cos, reduction='none')
        
        if exists(mel_lengths):
            mask = mask_from_frac_lengths(mel_lengths, *self.frac_lengths_mask)
            spec_loss = spec_loss * mask
            phase_loss = phase_loss * mask
            
        spec_loss = spec_loss.mean()
        phase_loss = phase_loss.mean()

        # 使用预测重建音频
        pred_audio = self.istft(pred_magnitude, pred_phase_sin, pred_phase_cos)
        gt_audio = self.istft(log_magnitude, phase_sin, phase_cos)
        
        # 计算重建损失
        recon_loss = F.l1_loss(pred_audio, gt_audio)

        # 总损失
        total_loss = spec_loss + phase_loss + 0.5 * recon_loss

        return total_loss
