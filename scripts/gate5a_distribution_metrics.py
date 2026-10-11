#!/usr/bin/env python3
"""Gate 5a distribution extensions that preserve the Gate 4b metric path."""

from __future__ import annotations

import torch

from gate4b_probabilistic_metrics import (
    LEVELS,
    central_intervals as gate4b_central_intervals,
    distribution as gate4b_distribution,
    empirical_crps as gate4b_empirical_crps,
    interval_metrics,
    nll_elements as gate4b_nll_elements,
)


FIXED_STUDENT_VARIANT = "student_t_fixed_df8_head"
LAPLACE_VARIANT = "laplace_head"


def distribution(variant, loc, scale, df=None):
    if variant == LAPLACE_VARIANT:
        return torch.distributions.Laplace(loc, scale)
    if variant == FIXED_STUDENT_VARIANT:
        return gate4b_distribution("student_t_head", loc, scale, df)
    return gate4b_distribution(variant, loc, scale, df)


def nll_elements(variant, loc, scale, target, df=None):
    if variant == LAPLACE_VARIANT:
        return torch.log(2.0 * scale) + torch.abs(target - loc) / scale
    if variant == FIXED_STUDENT_VARIANT:
        return gate4b_nll_elements("student_t_head", loc, scale, target, df)
    return gate4b_nll_elements(variant, loc, scale, target, df)


def laplace_quantile(probability, loc, scale):
    probability = torch.as_tensor(probability, device=loc.device, dtype=loc.dtype)
    centered = probability - 0.5
    return loc - scale * torch.sign(centered) * torch.log1p(-2.0 * torch.abs(centered))


def central_intervals(variant, loc, scale, df=None):
    if variant == LAPLACE_VARIANT:
        output = {}
        for level in LEVELS:
            lower_probability = (1.0 - level) / 2.0
            upper_probability = (1.0 + level) / 2.0
            output[level] = (
                laplace_quantile(lower_probability, loc, scale),
                laplace_quantile(upper_probability, loc, scale),
            )
        return output
    if variant == FIXED_STUDENT_VARIANT:
        return gate4b_central_intervals("student_t_head", loc, scale, df)
    return gate4b_central_intervals(variant, loc, scale, df)


def empirical_crps(variant, loc, scale, target, df=None, samples=64, seed=52025, chunk_elements=65536):
    if variant == FIXED_STUDENT_VARIANT:
        return gate4b_empirical_crps(
            "student_t_head", loc, scale, target, df, samples, seed, chunk_elements
        )
    if variant != LAPLACE_VARIANT:
        return gate4b_empirical_crps(variant, loc, scale, target, df, samples, seed, chunk_elements)

    flat_loc = loc.reshape(-1)
    flat_scale = scale.reshape(-1)
    flat_target = target.reshape(-1)
    total = 0.0
    count = 0
    devices = [loc.device.index] if loc.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for start in range(0, flat_loc.numel(), chunk_elements):
            chunk_loc = flat_loc[start : start + chunk_elements]
            chunk_scale = flat_scale[start : start + chunk_elements]
            chunk_target = flat_target[start : start + chunk_elements]
            samples_tensor = torch.distributions.Laplace(chunk_loc, chunk_scale).sample((samples,))
            first = (samples_tensor - chunk_target).abs().mean(0)
            sorted_samples = samples_tensor.sort(0).values
            coefficient = (
                2 * torch.arange(1, samples + 1, device=loc.device, dtype=loc.dtype) - samples - 1
            ).view(-1, 1)
            pair_half = (coefficient * sorted_samples).sum(0) / (samples * samples)
            crps = first - pair_half
            total += float(crps.sum())
            count += crps.numel()
    return total / count
